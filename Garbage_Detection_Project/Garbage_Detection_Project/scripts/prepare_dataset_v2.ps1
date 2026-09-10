[CmdletBinding()]
param(
    [string]$SourceRoot = (Join-Path $PSScriptRoot '..\dataset\train'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\dataset_yolo_pending_annotation'),
    [int]$Seed = 4050
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$classes = @('battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash')
$allowedExtensions = @('.jpg', '.jpeg', '.png')
$splitFractions = [ordered]@{ train = 0.70; val = 0.15; test = 0.15 }

$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)

if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) {
    throw "Original image root was not found: $SourceRoot"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to overwrite existing output: $OutputRoot"
}

Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class DatasetAuditFileLinks {
    [DllImport("Kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern bool CreateHardLink(string newFileName, string existingFileName, IntPtr securityAttributes);
}
'@

function Get-ImageIssue {
    param([string]$Path)
    try {
        $image = [System.Drawing.Image]::FromFile($Path)
        try {
            if ($image.Width -lt 2 -or $image.Height -lt 2) { return "invalid_dimensions" }
            return $null
        }
        finally { $image.Dispose() }
    }
    catch { return $_.Exception.Message }
}

function Get-DeterministicShuffle {
    param([array]$Items, [int]$RandomSeed)
    $result = [System.Collections.Generic.List[object]]::new()
    foreach ($item in $Items) { [void]$result.Add($item) }
    $rng = [System.Random]::new($RandomSeed)
    for ($i = $result.Count - 1; $i -gt 0; $i--) {
        $j = $rng.Next($i + 1)
        $tmp = $result[$i]
        $result[$i] = $result[$j]
        $result[$j] = $tmp
    }
    return $result.ToArray()
}

function New-NonDestructiveImageLink {
    param([string]$Source, [string]$Destination)
    try {
        if ([DatasetAuditFileLinks]::CreateHardLink($Destination, $Source, [IntPtr]::Zero)) {
            return 'hardlink'
        }
        throw "CreateHardLink failed with Windows error $([Runtime.InteropServices.Marshal]::GetLastWin32Error())."
    }
    catch {
        Copy-Item -LiteralPath $Source -Destination $Destination -ErrorAction Stop
        return 'copy'
    }
}

$allFolders = Get-ChildItem -LiteralPath $SourceRoot -Directory | Sort-Object Name
$foundFolders = @($allFolders.Name)
$missingClasses = @($classes | Where-Object { $_ -notin $foundFolders })
$unexpectedFolders = @($foundFolders | Where-Object { $_ -notin ($classes + 'clean') })

$records = [System.Collections.Generic.List[object]]::new()
$corrupt = [System.Collections.Generic.List[object]]::new()
$classSummary = [System.Collections.Generic.List[object]]::new()

foreach ($folder in $allFolders) {
    $category = $folder.Name
    $images = @(Get-ChildItem -LiteralPath $folder.FullName -File | Where-Object { $_.Extension.ToLowerInvariant() -in $allowedExtensions } | Sort-Object Name)
    $valid = [System.Collections.Generic.List[object]]::new()
    foreach ($image in $images) {
        $issue = Get-ImageIssue -Path $image.FullName
        if ($null -eq $issue) {
            [void]$valid.Add($image)
        }
        else {
            [void]$corrupt.Add([PSCustomObject]@{ source_path = $image.FullName; source_category = $category; issue = $issue })
        }
    }
    [void]$classSummary.Add([PSCustomObject]@{
        source_category = $category
        image_files = $images.Count
        valid_images = $valid.Count
        corrupt_or_unreadable = $images.Count - $valid.Count
        role = if ($category -eq 'clean') { 'negative/background review only' } elseif ($category -in $classes) { 'manual bounding-box annotation required' } else { 'excluded from v2' }
    })

    if ($category -notin ($classes + 'clean')) { continue }
    $categorySeed = $Seed + [Math]::Abs($category.GetHashCode())
    $shuffled = Get-DeterministicShuffle -Items $valid.ToArray() -RandomSeed $categorySeed
    $n = $shuffled.Count
    $trainCount = [Math]::Floor($n * $splitFractions.train)
    $valCount = [Math]::Floor($n * $splitFractions.val)
    $testCount = $n - $trainCount - $valCount
    for ($index = 0; $index -lt $n; $index++) {
        $split = if ($index -lt $trainCount) { 'train' } elseif ($index -lt ($trainCount + $valCount)) { 'val' } else { 'test' }
        $image = $shuffled[$index]
        # Include the source extension in the stem so .jpg/.png files with the
        # same original basename cannot collide with one YOLO label filename.
        $safeName = '{0}__{1}__{2}{3}' -f $category, $image.BaseName, $image.Extension.TrimStart('.'), $image.Extension
        [void]$records.Add([PSCustomObject]@{
            source_path = $image.FullName
            source_category = $category
            split = $split
            target_image_name = $safeName
            annotation_required = ($category -ne 'clean')
            annotation_status = if ($category -eq 'clean') { 'review_as_negative' } else { 'pending_manual_bbox_annotation' }
        })
    }
}

New-Item -ItemType Directory -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Path (Join-Path $OutputRoot 'manifests') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $OutputRoot 'reports') | Out-Null
foreach ($split in $splitFractions.Keys) {
    New-Item -ItemType Directory -Path (Join-Path $OutputRoot "images\\$split") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $OutputRoot "labels\\$split") -Force | Out-Null
}

$linkCounts = @{ hardlink = 0; copy = 0 }
foreach ($record in $records) {
    $targetImage = Join-Path $OutputRoot "images\\$($record.split)\\$($record.target_image_name)"
    $targetLabel = Join-Path $OutputRoot "labels\\$($record.split)\\$([IO.Path]::GetFileNameWithoutExtension($record.target_image_name)).txt"
    $method = New-NonDestructiveImageLink -Source $record.source_path -Destination $targetImage
    $linkCounts[$method]++
    New-Item -ItemType File -Path $targetLabel | Out-Null
}

$records | Export-Csv -NoTypeInformation -Encoding utf8 -Path (Join-Path $OutputRoot 'manifests\\image_manifest.csv')
$records | Where-Object annotation_required | Select-Object source_path, source_category, split, target_image_name, annotation_status | Export-Csv -NoTypeInformation -Encoding utf8 -Path (Join-Path $OutputRoot 'manifests\\annotation_tasks.csv')
$classSummary | Export-Csv -NoTypeInformation -Encoding utf8 -Path (Join-Path $OutputRoot 'reports\\class_distribution.csv')
$corrupt | Export-Csv -NoTypeInformation -Encoding utf8 -Path (Join-Path $OutputRoot 'reports\\corrupt_or_unreadable_images.csv')

$dataYaml = @"
path: $OutputRoot
train: images/train
val: images/val
test: images/test
names:
  0: battery
  1: biological
  2: cardboard
  3: clothes
  4: glass
  5: metal
  6: paper
  7: plastic
  8: shoes
  9: trash
"@
Set-Content -LiteralPath (Join-Path $OutputRoot 'data.yaml') -Value $dataYaml -Encoding utf8

$report = [ordered]@{
    generated_at = (Get-Date).ToString('o')
    source_root = $SourceRoot
    output_root = $OutputRoot
    classes = $classes
    excluded_detection_class = 'clean'
    source_folders_found = $foundFolders
    missing_required_folders = $missingClasses
    unexpected_source_folders = $unexpectedFolders
    split_fractions = $splitFractions
    total_valid_images = $records.Count
    images_requiring_manual_bbox_annotation = @($records | Where-Object annotation_required).Count
    clean_images_requiring_negative_review = @($records | Where-Object { $_.source_category -eq 'clean' }).Count
    corrupt_or_unreadable_images = $corrupt.Count
    materialization = $linkCounts
    annotation_gate = 'DO NOT TRAIN: every garbage image has an intentionally empty placeholder label pending manual bounding-box annotation.'
    class_distribution = @($classSummary)
}
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputRoot 'reports\\audit_report.json') -Encoding utf8

$instructions = @"
# Annotation gate

`dataset_v2` deliberately contains no inherited labels from `dataset/labels/train`.
Those labels were generated from a pretrained COCO detector and are invalid for this ten-class project.

Annotate only these classes, with these exact IDs: battery=0, biological=1, cardboard=2, clothes=3, glass=4, metal=5, paper=6, plastic=7, shoes=8, trash=9.

For each garbage image, create a YOLO `.txt` file of one row per visible garbage object:
`class_id x_center y_center width height`, all coordinates normalized from 0 to 1.

Review every `clean` image. Keep its matching label file empty only if no target garbage object is visible. If target garbage is present, move it into the appropriate annotation workflow and add correct boxes.

Do not train until annotation review and the final label audit report both pass.
"@
Set-Content -LiteralPath (Join-Path $OutputRoot 'ANNOTATION_REQUIRED.md') -Value $instructions -Encoding utf8

Write-Host "Prepared non-destructive dataset_v2 at $OutputRoot"
Write-Host "Valid images: $($records.Count); manual annotations required: $(@($records | Where-Object annotation_required).Count); corrupt/unreadable: $($corrupt.Count)"

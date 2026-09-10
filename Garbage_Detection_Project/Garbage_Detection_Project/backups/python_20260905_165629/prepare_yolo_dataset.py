import os
import shutil

source_root = "dataset/train"
destination = "dataset/images/train"

os.makedirs(destination, exist_ok=True)

for folder in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                src_file = os.path.join(folder_path, file)
                dst_file = os.path.join(destination, file)

                shutil.copy(src_file, dst_file)

print("✅ All images moved to dataset/images/train")
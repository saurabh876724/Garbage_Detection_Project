import os
import shutil
import random

train_path = r"C:\Users\Saurabh\My All Projects\Garbage_Detection_Project\dataset\train\garbage"
val_path = r"C:\Users\Saurabh\My All Projects\Garbage_Detection_Project\dataset\validation\garbage"

split_ratio = 0.15  # 15% images move to validation

for class_name in os.listdir(train_path):
    class_train_folder = os.path.join(train_path, class_name)
    class_val_folder = os.path.join(val_path, class_name)

    os.makedirs(class_val_folder, exist_ok=True)

    images = os.listdir(class_train_folder)
    random.shuffle(images)

    split_count = int(len(images) * split_ratio)
    val_images = images[:split_count]

    for img in val_images:
        src = os.path.join(class_train_folder, img)
        dst = os.path.join(class_val_folder, img)
        shutil.move(src, dst)

print("✅ Validation images moved successfully!")

import os
import shutil

source_folder = 'ds_code/Datasets/Test/Negative'
destination_folder = 'ds_code/Datasets/Validation/Negative'
interval = 2  # Number of images to skip between transfers

image_files = os.listdir(source_folder)
count = 0

for filename in image_files:
    if count % interval == 0:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.move(source_path, destination_path)
    count += 1
print("Images Transferred Successfully....'")
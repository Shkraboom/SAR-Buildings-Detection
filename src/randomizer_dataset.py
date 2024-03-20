import os
import random
import shutil

random.seed(42)

images_folder = "path_to_images_folder"
labels_folder = "path_to_labels_folder"

train_folder = "path_to_train_folder"
validation_folder = "path_to_validation_folder"
test_folder = "path_to_test_folder"

os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)
os.makedirs(os.path.join(validation_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(validation_folder, 'labels'), exist_ok=True)
os.makedirs(os.path.join(test_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_folder, 'labels'), exist_ok=True)

images_files = os.listdir(images_folder)
labels_files = os.listdir(labels_folder)

random.shuffle(images_files)

train_size = int(0.7 * len(images_files))
validation_size = int(0.2 * len(images_files))
test_size = len(images_files) - train_size - validation_size

train_images = images_files[:train_size]
validation_images = images_files[train_size:train_size + validation_size]
test_images = images_files[train_size + validation_size:]

def copy_annotations(image_files, src_folder, dest_folder):
    for image_file in image_files:
        label_file = image_file.replace('.jpg', '.txt')
        shutil.copy(os.path.join(src_folder, label_file), os.path.join(dest_folder, 'labels'))

copy_annotations(train_images, labels_folder, train_folder)
copy_annotations(validation_images, labels_folder, validation_folder)
copy_annotations(test_images, labels_folder, test_folder)

for image_file in train_images:
    shutil.copy(os.path.join(images_folder, image_file), os.path.join(train_folder, 'images'))

for image_file in validation_images:
    shutil.copy(os.path.join(images_folder, image_file), os.path.join(validation_folder, 'images'))

for image_file in test_images:
    shutil.copy(os.path.join(images_folder, image_file), os.path.join(test_folder, 'images'))
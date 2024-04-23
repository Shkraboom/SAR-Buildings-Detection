import cv2
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import shutil

class Model(object):
    def __init__(self, model_path):
        self.yolo_model = YOLO(model_path)
    def split_and_save_image(self, image_path, chunk_size = 512):
        """
        Кроп снимков и сохранение фрагментов
        """
        self.image_path = image_path
        self.chunk_size = chunk_size

        img = Image.open(image_path)

        cwd = os.getcwd()
        new_dir = 'output_folder'
        path = os.path.join(cwd, new_dir)

        shutil.rmtree(path, ignore_errors = True)
        os.makedirs(path, exist_ok = True)

        nump_chunks = []
        width, height = img.size

        coords = {'x':0, 'y':0}

        coords['x'] = width // chunk_size
        coords['y'] = height // chunk_size

        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                chunk = img.crop((x, y, x + chunk_size, y + chunk_size))

                chunk_array = np.array(chunk)
                nump_chunks.append(chunk_array)

                chunk_filename = f"{path}/chunk_{x}_{y}.jpg"
                Image.fromarray(chunk_array).save(chunk_filename)
        return coords

    def speckle_denoising(self, image_path, output_folder, window_size = 20):
        """
        Фильтрация speckle-шума
        """
        self.image_path = image_path
        self.window_size = window_size

        img = cv2.imread(image_path)
        img_filtered = cv2.fastNlMeansDenoising(img, None, h = window_size)

        output_filename = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_filename, img_filtered)
        return output_filename

    def merge_images(self, image_folder, coords, save = True, chunk_size = 512):
        """
        Склейка снимка до исходного изображения
        """
        self.image_folder = image_folder
        self.coords = coords
        self.save = save
        self.chunk_size = chunk_size

        num_chunks_horizontal = coords['x']
        num_chunks_vertical = coords['y']

        image_rows = []

        for y in range(num_chunks_vertical):
            row_images = []
            for x in range(num_chunks_horizontal):
                filename = f"chunk_{x * chunk_size}_{y * chunk_size}.jpg"
                filepath = os.path.join(image_folder, filename)

                image = cv2.imread(filepath, cv2.COLOR_BGR2RGB)

                row_images.append(image)

            row_merged = np.hstack(row_images)

            image_rows.append(row_merged)

        merged_image = np.vstack(image_rows)

        if save:
            output_folder = os.path.join(os.getcwd(), 'output_merge_folder')
            shutil.rmtree(output_folder, ignore_errors = True)
            os.makedirs(output_folder, exist_ok = True)

            output_filename = os.path.join(output_folder, 'merged_image.jpg')

            cv2.imwrite(output_filename, merged_image)

            return merged_image
        else:
            return merged_image

    def predict(self, image_path, chunk_size = 512, filtered = False, window_size = 20):
        """
        Предсказание на снимке
        """
        self.image_path = image_path
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.filtered = filtered

        output_yolo_folder = os.path.join(os.getcwd(), 'output_yolo_folder')
        shutil.rmtree(output_yolo_folder, ignore_errors = True)
        os.makedirs(output_yolo_folder, exist_ok = True)

        coords = self.split_and_save_image(image_path = image_path, chunk_size = chunk_size)

        if filtered:
            output_folder = 'output_folder'

            for filename in os.listdir(output_folder):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(output_folder, filename)
                    self.speckle_denoising(image_path, output_folder, window_size = window_size)
            self.yolo_model(source=output_folder, save = True, save_txt = True, show_labels = False, project = output_yolo_folder,
                       name='predict')
            pred = self.merge_images(image_folder=os.path.join(output_yolo_folder, 'predict'), coords = coords, save = True)
        else:
            output_folder = 'output_folder'
            self.yolo_model(source = output_folder, save = True, save_txt = True, show_labels = False, project = output_yolo_folder, name = 'predict')
            pred = self.merge_images(image_folder = os.path.join(output_yolo_folder, 'predict'), coords = coords, save = True, chunk_size = chunk_size)

        return pred
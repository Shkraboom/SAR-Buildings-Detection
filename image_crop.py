from PIL import Image
import numpy as np
import os

def split_and_save_image(image_path, output_folder):
    img = Image.open(image_path)

    width, height = img.size

    chunk_size = 512

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nump_chunks = []

    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            chunk = img.crop((x, y, x + chunk_size, y + chunk_size))

            chunk_array = np.array(chunk)
            nump_chunks.append(chunk_array)

            chunk_filename = f"{output_folder}/chunk_{x}_{y}.jpg"
            Image.fromarray(chunk_array).save(chunk_filename)

image_path = "path_to_input_image"
output_folder = "path_to_output_folder"

split_and_save_image(image_path, output_folder)

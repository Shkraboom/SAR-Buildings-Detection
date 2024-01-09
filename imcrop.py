import torch
from PIL import Image
import numpy as np
import os

def split_and_save_image(image_path, output_folder, tens_folder):
    img = Image.open(image_path)

    width, height = img.size

    chunk_size = 512

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(tens_folder):
        os.makedirs(tens_folder)

    tens_chunks = []
    nump_chunks = []

    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            chunk = img.crop((x, y, x + chunk_size, y + chunk_size))

            chunk_array = np.array(chunk)
            tens_array = torch.from_numpy(chunk_array)

            tens_chunks.append(tens_array)
            nump_chunks.append(chunk_array)

            chunk_filename = f"{output_folder}/chunk_{x}_{y}.jpg"
            Image.fromarray(chunk_array).save(chunk_filename)

    torch.save(tens_chunks, os.path.join(tens_folder, "tens_chunks.pt"))

image_path = "/Users/daniilskrabo/Desktop/images/sar_0.jpg"
output_folder = "/Users/daniilskrabo/Desktop/output/jpg_chunks"
tens_folder = "/Users/daniilskrabo/Desktop/output/tensor_chunks"
split_and_save_image(image_path, output_folder, tens_folder)
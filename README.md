# Building detection on SAR images
This project implements the YOLOv8 neural network architecture for building detection on SAR (synthesised aperture radar) images.

## Table of Contents
- [Project Description](#project-description)
- [Technologies](#technologies)
- [Data preprocessing](#data-preprocessing)
- [Model Training](#training-model)
- [Model class implementation](#implementation-class-model)
- [Docker Image](#docker-image)
- [FastAPI application](#fastapi-application)
- [Final Result](#final-result)

## Project Description
This project implements YOLOv8 neural network architecture for building detection in SAR images. SAR images are radar images of the Earth obtained during the reflection of radar waves from its surface. The advantage of such images is their independence from meteorological conditions and time of day. 
YOLOv8 OBB (oriented bounding box) neural network was chosen as a detector. The OBB function helps to determine the location of buildings and their coordinates more accurately, as they are often rotated at an indirect angle.

## Technologies
- [YOLOv8](https://docs.ultralytics.com/)
- [Python](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [CVAT](https://www.cvat.ai/)

## Data preprocessing
One of the goals of this project was to be able to apply the model to images of different resolutions. However, the model is trained on samples of the same resolution. I set this resolution to 512 x 512 pixels.
For efficient partitioning, I wrote a method `split_and_save_image`, which divides the image into semls of the specified resolution. I did the partitioning in CVAT. In this software you can rotate bounding boxes to a certain angle, which is required for my project.
However, due to frequent blurring of PCA images themselves, the marking could not be done - buildings are not visible. That's why I made additional labelling of buildings using [OpenStreetMap](https://www.openstreetmap.org/#map=13/-33.4377/-70.7966&layers=N) and its API [Overpass API](https://overpass-turbo.eu/).

![labelling.jpg](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/metrics/labeling.jpg)

Next, I wrote a `randomiser_dataset` script to break down the training, validation and test samples.

## Training the model
I used Ultralytics YOLOv8 nano OBB as the neural network architecture. Training took place with the following hyperparameters:
- `epochs` = 100
- `imgsz` = 512
- `batch` = 16
- `iou` = 0.7

Despite the small number of weights and relatively small dataset (only 500 samples), the model showed good metrics: 

![confusion_matrix_normalised.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/train64_base/confusion_matrix_normalized.png)
![results.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/train64_base/results.png)
![F1_curve.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/train64_base/F1_curve.png)

Results of running the model on the validation sample:

![val_batch0_pred.jpg](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/train64_base/val_batch0_pred.jpg)

## Implementation of the Model class
After training, it was necessary to build the whole project. To do this, I wrote a Model class with methods such as:
- `split_and_save_image` - makes a crop of the input image (image_path) with the given size (chunk_size)
- `speckle_denoising` - performs speckle noise filtering using cv2::fastNLMeansDenoising filter with specified window size
- `merge_images` - merges samples to the original resolution
- `predict` - merges the previous methods and makes predicates using a trained YOLOv8 Nano OBB model

## Docker image
I also created an image in Docker and uploaded it to [DockerHub](https://hub.docker.com/repository/docker/shkraboom/sar/general):

`docker pull shkraboom/sar` - upload the image

`docker run -it -v path/to/images/folder:/app/images sar /bin/bash` - run the image in the container, and create the snapshot directory as well

`root@<container_id>:/app# python3`.

`>>> from model import Model`

`>>>> model = Model(model_path = '/app/data/train64_base/weights/last.pt')`

`>>>> model.predict(image_path = '/app/images/your_image.jpg', chunk_size = 1024, filtered = False, window_size = 20)`

## FastAPI application
Docker image of the FastAPI application also posted on [DockerHub](https://hub.docker.com/repository/docker/shkraboom/sar_api/general):

`docker run -p 8000:8000 shkraboom/sar_api` - run the image

The `predict` endpoint takes a snapshot and returns the URL of the input and output snapshot. Endpoints `/images/predictions/{filename}` and `/images/originals/{filename}` open the output and input snapshot respectively. 

## Final result
After applying the `predict` method, the result of the model is saved in the `output_merge_folder` folder. The test example is a 11264 x 10404 pixel image:

![test_image.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/metrics/test_image.png)
![predict_test_image.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/metrics/predict_test_image.png)
![predict_test_image_2.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/metrics/predict_test_image_2.png)
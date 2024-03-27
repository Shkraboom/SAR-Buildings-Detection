# Детекция зданий на РСА снимках
В этом проекте реализована архитектура нейронной сети YOLOv8 для детекции зданий на РСА (радиолокационных синтезированных апертурах) снимках.

## Содержание
- [Описание проекта](#описаниепроекта)
- [Технологии](#технологии)
- [Предобработка данных](#предобработка-данных)
- [Обучение модели](#обучение-модели)
- [Реализация класса Model](#реализация-класса-model)
- [Образ Docker](#образ-docker)
- [Финальный результат](#финальный-результат)

## Описание проекта
В данном проекте реализована архитектура нейронной сети YOLOv8 для детекции зданий на РСА снимках. РСА (SAR) снимки - это радиолокационные изображения Земли, получаемые в ходе отражения радиолокационных волн от её поверхности. Приемущество изображений такого формата - независимость от метеоусловий и времени суток. 
В качестве детектора была выбрана нейронная сеть YOLOv8 OBB (oriented bounding box). Функция OBB помогает более точно определять местоположение зданий и их координаты, так как зачастую они повернуты на непрямой угол.

## Технологии
- [YOLOv8](https://docs.ultralytics.com/)
- [Python](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [CVAT](https://www.cvat.ai/)

## Предобработка данных
Одной из задач данного проекта являлась возможность применять модель к снимкам разного разрешения. Однако тренируется модель на семлах одинакового разрешения. Это разрешение я установил равным 512 x 512 пикселей.
Для эффективной разметки я написал метод 'split_and_save_image', который делит изображение на семлы заданного разрешения. Разметку я производил в CVAT. В этом софте можно поворачивать bounding box'ы на определенный угол, что требуется для моего проекта.
Однако из-за частой смазанности самих РСА снимков, разметку проводить не удавалось - здания не видны. Поэтому я сделал дополнительную разметку зданий с помощью [OpenStreetMap](https://www.openstreetmap.org/#map=13/-33.4377/-70.7966&layers=N) и его API [Overpass API](https://overpass-turbo.eu/).

![labeling.jpg](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/labeling.jpg)

Далее для разбивки на тренировочную, валидационную и тестовую выборки я написал скрипт 'randomizer_dataset'.

## Обучение модели
В качестве архитектуры нейронной сети я использовал Ultralytics YOLOv8 nano OBB. Обучение происходило с такими гиперпараметрами:
- epochs = 100
- imgsz = 512
- batch = 16
- iou = 0.7

Несмотря на малое количество весов и сравнительно небольшой датасет (всего 500 семплов), модель показала неплохие метрики: 

![confusion_matrix_normalized.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/train64_base/confusion_matrix_normalized.png)
![results.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/train64_base/results.png)
![F1_curve.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/train64_base/F1_curve.png)

Результаты работы модели на валидационной выборке:

![val_batch0_pred.jpg](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/train64_base/val_batch0_pred.jpg)

## Реализация класса Model
После обучения нужно было собрать проект целиком. Для этого я написал класс Model с такими методами:
- split_and_save_image - делает кроп входного изображения (image_path) с заданными размером (chunk_size)
- speckle_denoising - производит фильтрацию speckle шума с помощью фильтра cv2::fastNLMeansDenoising с заданным размером окна (window size)
- merge_images - склеивает семплы до исходного разрешения
- predict - объединяет предыдущие методы и делает предикты с помощью обученной модели YOLOv8 Nano OBB

## Образ Docker
Также я создал образ в Docker и выложил его в [DockerHub](https://hub.docker.com/repository/docker/shkraboom/sar/general):

`docker pull shkraboom/sar` — загружаем образ

`docker run -it -v path/to/images/folder:/app/images sar /bin/bash` — запускаем образ в контейнере, а также создаем каталог со снимками

`root@<container_id>:/app# python3`

`>>> from model import Model`

`>>> model = Model(model_path = '/app/data/train64_base/weights/last.pt')`

`>>> model.predict(image_path = '/app/images/your_image.jpg', chunk_size = 1024, filtered = False, window_size = 20)`

## Финальный результат
После применения метода 'predict' результат работы модели сохраняется в папке 'output_merge_folder'. Тестовый пример - снимок 11264 х 10404 пикселей:

![test_image.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/test_image.png)
![predict_test_image.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/predict_test_image.png)
![predict_test_image_2.png](https://github.com/Shkraboom/SAR-Buildings-Detection/blob/main/data/predict_test_image_2.png)
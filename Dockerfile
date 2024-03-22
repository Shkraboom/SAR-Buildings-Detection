FROM ubuntu:latest

LABEL authors="daniilskrabo"

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

RUN apt-get install -y libgl1-mesa-glx

RUN apt-get install -y libglib2.0-0


CMD ["python3", "src/model.py"]

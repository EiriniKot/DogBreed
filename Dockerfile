FROM python:3.10.5-slim-buster

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ENV TRAIN_DIR=datasets/yolov8_imgs/train
ENV VAL_DIR=datasets/yolov8_imgs/val
ENV TEST_DIR=datasets/yolov8_imgs/test
ENV DATA_DIR=datasets/Dataset/Images

COPY requirements.txt requirements.txt
COPY main_results.py main_results.py
COPY main_train.py main_train.py
COPY tools.py tools.py

COPY runs runs
COPY datasets datasets

RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]

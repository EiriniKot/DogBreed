FROM ubuntu:22.04
LABEL authors="eirini"

FROM python:3.10.5-slim-buster

ENV TRAIN_DIR=datasets/yolov8_imgs/train
ENV VAL_DIR=datasets/yolov8_imgs/val
ENV TEST_DIR=datasets/yolov8_imgs/test
ENV DATA_DIR=datasets/Dataset/Images

COPY requirements.txt requirements.txt
COPY main.py main.py
COPY runs runs
COPY datasets datasets
COPY dogbreed.yaml dogbreed.yaml


RUN pip install --no-cache-dir -r requirements.txt

#SHELL ["conda","run","-n","dogbreed","/bin/bash","-c"]

CMD ["/bin/bash"]
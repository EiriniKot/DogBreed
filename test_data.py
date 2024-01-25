import os
import cv2
import torch
from ultralytics import YOLO

test_set = "datasets/yolov8_imgs/test/images"
label_test_set = "datasets/yolov8_imgs/test/labels"
model = YOLO('runs/detect/yolov8n_dogbreed/weights/best.pt')
print(f'Task : {model.task}')

conf_threshold = 0.3
# results =

for img in os.listdir(test_set):
    img_ = cv2.imread(os.path.join(test_set,img))
    output = model.predict(img_, save_dir = "datasets/yolov8_imgs/test_out")[0]
    predicted_breeds = output.boxes.cls
    filtered_predictions = torch.where(output.boxes.conf>conf_threshold)

    full_target = open(os.path.join(label_test_set, img.split('.')[0]+'.txt'), "r").readlines()
    target_label = [line.split(" ")[0] for line in full_target]

    for pred in output[0].boxes:
            pass
    # ggggg


# Train the model
# results = model.train(data='dogbreed.yaml',
#                       epochs=24,
#                       imgsz=240,
#                       batch=30,
#                       name='yolov8n_dogbreed')

import scipy.io
from ultralytics import YOLO
from tools import create_yolov8_dataset

mat = scipy.io.loadmat('data/train_data.mat')
train_info = mat['train_info']
train_files = train_info[0][0][0].flatten()

mat = scipy.io.loadmat('data/test_data.mat')
test_info = mat['test_info']
test_files = test_info[0][0][0].flatten()

# This is the faster solution where we handle the problem
#  as a simple classification problem ignoring annotations
create_yolov8_dataset(train_files,
                      test=test_files,
                      type_of_problem='obj-detection',
                      valid_size=0.2,
                      random_state=42)

# N stands for nano which is the smaller version of yolo
# yolo_sizes = ['l']
# for size in yolo_sizes:
#     model = YOLO(f'yolov8{size}-cls.pt')  # load a pretrained model (recommended for training)
#     # Train the model
#     results = model.train(data='data/yolov8_imgs',
#                           imgsz=220,
#                           epochs=3,
#                           batch=50,
#                           name='yolov8n_custom')

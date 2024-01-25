import scipy.io
from ultralytics import YOLO
from tools import create_yolov8_dataset

mat = scipy.io.loadmat('datasets/train_data.mat')
train_info = mat['train_info']
train_files = train_info[0][0][0].flatten()
mat = scipy.io.loadmat('datasets/test_data.mat')
test_info = mat['test_info']
test_files = test_info[0][0][0].flatten()

create_yolov8_dataset(train_files,
                      test=test_files,
                      name='dogbreed.yaml',
                      task='detect',
                      valid_size=0.2,
                      random_state=42)

# N stands for nano which is the smaller version of yolo
yolo_sizes = ['m']
for size in yolo_sizes:
    # load a pretrained model (recommended for training)
    model = YOLO(f'yolov8{size}.pt', task='detect')
    # Train the model
    results = model.train(data='dogbreed.yaml',
                          epochs=100,
                          imgsz=220,
                          batch=64,  # For batch we pick a power of two that my laptop can handle
                          name=f'yolov8n_dogbreed_2501_{size}')

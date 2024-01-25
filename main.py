import os
import scipy.io
from ultralytics import YOLO
from tools import create_yolov8_dataset

mat = scipy.io.loadmat('datasets/train_data.mat')
train_info = mat['train_info']
train_files = train_info[0][0][0].flatten()
mat = scipy.io.loadmat('datasets/test_data.mat')
test_info = mat['test_info']
test_files = test_info[0][0][0].flatten()

make_single_dog_label = True
# Find what classes are available
classes = [lbl.split("-", 1)[1] for lbl in os.listdir(os.environ['DATA_DIR'])]

if make_single_dog_label==True:
    # If we go with solution number 3. We will flatten classes into just dog.
    classes = ['Dog']

create_yolov8_dataset(train_files,
                      classes=classes,
                      test=test_files,
                      name='dogbreed.yaml',
                      task='detect',
                      valid_size=0.2,
                      random_state=42)

# n = nano which is the smaller version of yolo
yolo_sizes = ['m']
for size in yolo_sizes:
    # load a pretrained model (recommended for training)
    model = YOLO(f'yolov8{size}.pt', task='detect')
    # Train the model
    results = model.train(data='dogbreed.yaml',
                          epochs=50,
                          imgsz=180,
                          batch=32,  # (For batch we pick a power of two)
                          patience=15,
                          name=f'yolov8n_dogbreed_2501_{size}')

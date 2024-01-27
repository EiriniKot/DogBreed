import os, shutil, yaml
import numpy as np
from typing import Optional
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from PIL import Image
from ultralytics import YOLO

class Trainer:
    def __init__(self,
                 train_files,
                 test_files,
                 data_directory,
                 valid_size: float = 0.2,
                 random_state: int = 42,
                 yolo_sizes: list = ['n'],
                 model_kwargs: dict = {'epochs': 2,
                                       'imgsz': 22,
                                       'batch': 32,  # (For batch we pick a power of two)
                                       'patience': 15},
                 save=True):
        # Split datasets once.
        split_train_val(train_files, valid_size=valid_size, random_state=random_state, save=save)
        # self.train_files = train_files
        self.test_files = test_files
        self.data_directory = data_directory
        self.yolo_sizes = yolo_sizes
        self.model_kwargs = model_kwargs

    def prepare_solution(self, solution):
        if solution in ['SOLUTION_1', 'solution_1']:
            # Find what classes are available
            self.classes = [lbl.split("-", 1)[1] for lbl in os.listdir(self.data_directory)]
            self.task = 'classify'
            # All classification available models in yolo end with -cls
            self.yolo_type = '-cls'
            self.data = 'datasets/yolov8_imgs'
        elif solution in ['SOLUTION_2', 'solution_2']:
            # Find what classes are available
            self.classes = [lbl.split('-', 1)[1] for lbl in os.listdir(os.environ['DATA_DIR'])]
            self.task = 'detect'
            self.yolo_type = ''
            self.data = f'dogbreed.yaml'
        else:
            # single_cls = True

            raise ValueError('Wrong solution name')

    def train(self, solution):
        self.prepare_solution(solution)
        create_yolov8_dataset(classes=self.classes,
                              # Get paths for train and validation set to save the dataset
                              train_dir=os.environ['TRAIN_DIR'],
                              val_dir=os.environ['VAL_DIR'],
                              dataset_initial_dir=self.data_directory,
                              test=self.test_files,
                              name=f'dogbreed.yaml',
                              task=self.task)

        all_results = []
        for model_size in self.yolo_sizes:
            # load a pretrained model (recommended for training)
            print(f'Task: {self.task}')
            model = YOLO(f'yolov8{model_size}{self.yolo_type}.pt', task=self.task)
            # Train the model
            results = model.train(data=self.data,
                                  name=f'yolov8_{solution}_{self.task}_{model_size}',
                                  **self.model_kwargs)
            all_results.append(results)
            del (model)
            print('Remove yolo dataset directory.')
        shutil.rmtree('datasets/yolov8_imgs')
        return all_results


def convert(size, box):
    """
    Converts bounding box coordinates from pixel values to normalized values (for yolo annotation),
    :param size: A tuple representing the width and height of the image in pixels.
    :param box: A tuple representing the coordinates of the bounding box in pixel values.
    :return:  The converted bounding box coordinates in normalized values (x, y, w, h).
    """
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x, y, w, h


def convert_annotation(dir_path, output_path, classes):
    """
    This function is responsible for converting labels of ( PASCAL Visual Object Classes (VOC) )
     to yolo compatible annotations
    :param dir_path: str : Single annotation file path (annotation file can be in .txt)
    :param output_path: str : The output directory to save the yolo label.
    :param classes: lst : list[str] of classes
    :return:
    """
    basename = os.path.basename(dir_path)
    os.makedirs(output_path, exist_ok=True)

    out_file = open(output_path + basename + '.txt', 'w')
    tree = ET.parse(dir_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        if classes == ['Dog']:
            cls_id = 0
        else:
            cls = obj.find('name').text
            cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def copy_img(dir, src_img, dataset_initial_dir, task):
    if task == "detect":
        intermediate_lbl = ''
    else:
        intermediate_lbl = os.path.dirname(src_img).split('-', 1)[1]
    lbl = os.path.join(dir, intermediate_lbl)
    os.makedirs(lbl, exist_ok=True)
    new_path = os.path.join(lbl, os.path.basename(src_img))
    shutil.copy(os.path.join(dataset_initial_dir, src_img), new_path)


def create_yaml(train_dir: str,
                valid_dir: str,
                classes,
                test_dir: Optional[str] = None,
                name='yolo_dataset.yml'):

    yaml_data = {
        'path': os.getcwd(),
        'train': train_dir,
        'val': valid_dir,
        'test': test_dir,
        'names': {idx: class_name for idx, class_name in enumerate(classes)}
    }

    with open(name, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)


def split_train_val(image_paths, valid_size: float = 0.2, random_state: int = 42, save=True):
    assert (0 <= valid_size < 1), "Invalid valid size for yolov8 setup"
    breed_lbl = [os.path.dirname(pth[0]).split('-', 1)[1] for pth in image_paths]
    # Split the dataset into training and validation sets in a stratified manner
    train_paths, val_paths = train_test_split(image_paths,
                                                stratify=breed_lbl,
                                                test_size=valid_size,
                                                train_size=1 - valid_size,
                                                random_state=random_state)
    if save:
        np.save('train_sample_ids.npy', train_paths)
        np.save('val_sample_ids.npy', val_paths)

    print("Dataset split into training and valation sets.")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print("-"*50)
    return train_paths, val_paths


def create_yolov8_dataset(classes: list[str],
                          train_dir:str,
                          val_dir:str,
                          dataset_initial_dir:str,
                          test: Optional[np.ndarray] = None,
                          name: str = 'yolo_dataset.yml',
                          task: str = 'detect'):
    """
    Create the expected yolov8 dataset format as presented in
    https: // docs.ultralytics.com / datasets / classify /
    :param classes: List of available classes for example ['Eskimo_dog', 'bull_mastiff', 'Ibizan_hound']
    :param test: Optional test path for train+valid set
    :param name: str Name of the yml file for passing into yolov8 only used in detect tasks
    :param task: One of 'detect', 'classify'
    :return:
    """
    assert task in ['detect', 'classify'], \
        "Type of problem should be 'detect' or 'classify'"

    train_paths = np.load('train_sample_ids.npy', allow_pickle=True)
    val_paths = np.load('val_sample_ids.npy', allow_pickle=True)

    # Copy images to the training directory
    for train_img in train_paths:
        copy_annotate_set(img=train_img[0],
                          set_dir=train_dir,
                          previous_dir=dataset_initial_dir,
                          classes=classes,
                          task=task)

    # Copy images to the validation directory
    for val_img in val_paths:
        copy_annotate_set(img=val_img[0],
                          set_dir=val_dir,
                          previous_dir=dataset_initial_dir,
                          classes=classes,
                          task=task)

    if test is not None:
        test_dir = os.environ['TEST_DIR']
        for test_img in test:
            copy_annotate_set(img=test_img[0],
                              set_dir=test_dir,
                              previous_dir=dataset_initial_dir,
                              classes=classes,
                              task=task)
        print(f"Test samples: {len(test)}")
    else:
        test_dir = None

    create_yaml(train_dir, val_dir, classes, test_dir, name=name)


def copy_annotate_set(img, set_dir, previous_dir, classes, task):
    if task == "detect":
        full_anno_path = os.path.join("datasets/Annotation", os.path.splitext(img)[0])
        convert_annotation(full_anno_path, f"{set_dir}/labels/", classes)
        img_new_dir = os.path.join(set_dir, "images")
    else:
        img_new_dir = set_dir
    copy_img(img_new_dir, img, previous_dir, task)


def crop_bbox():
    # Not Utilized Yet.
    # Get paths for train and validation set to save the dataset
    train_dir = os.environ['TRAIN_DIR']
    img_train_dir = os.path.join(train_dir, "images")
    ann_train_dir = os.path.join(train_dir, "labels")

    for img_name in os.listdir(img_train_dir):
        full_img_path = os.path.join(img_train_dir, img_name)
        im = Image.open(full_img_path)
        bboxes = open(ann_train_dir + img_name.split('.')[0] + '.txt', 'w')
        for bbox in bboxes:
            im = im.crop(bbox)
            im.save(full_img_path)


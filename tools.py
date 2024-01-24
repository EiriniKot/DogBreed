from typing import Optional
import xml.etree.ElementTree as ET
import os
import shutil

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def convert(size, box):
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
        cls = obj.find('name').text
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def copy_img(dir, src_img, dataset_initial_dir, task):
    """

    :param dir:
    :param src_img:
    :param dataset_initial_dir:
    :return:
    """
    if task=="detect":
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
        'path': "/home/eirini/PycharmProjects/DogBreed",
        'train': train_dir,
        'val': valid_dir,
        'test': test_dir,
        'names': {idx: class_name for idx, class_name in enumerate(classes)}
    }

    with open(name, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)


def create_yolov8_dataset(image_paths: np.ndarray,
                          test: Optional[np.ndarray] = None,
                          name: str = 'yolo_dataset.yml',
                          task: str = 'detect',
                          valid_size: float = 0.2, random_state: int = 42):
    """
    Create the expected yolov8 dataset format as presented in
    https: // docs.ultralytics.com / datasets / classify /
    :param image_paths: Image paths for train+valid set
    :param test: Optional test path for train+valid set
    :param name:
    :param task: One of 'detect', 'classify'
    :param valid_size: 0-1 range of validation size for validation set
    :param random_state:
    :return:
    """
    assert (0 <= valid_size < 1), "Invalid valid size for yolov8 setup"
    assert task in ['detect', 'classify'], \
        "Type of problem should be 'detect' or 'classify'"

    breed_lbl = [os.path.dirname(pth[0]).split('-', 1)[1] for pth in image_paths]
    # Split the dataset into training and validation sets in a stratified manner
    train_paths, valid_paths = train_test_split(image_paths,
                                                stratify=breed_lbl,
                                                test_size=valid_size,
                                                train_size=1-valid_size,
                                                random_state=random_state)

    # Create directories for training and validation sets
    train_dir = "datasets/yolov8_imgs/train"
    valid_dir = "datasets/yolov8_imgs/val"
    dataset_initial_dir = "datasets/Dataset/Images"
    img_train_dir = os.path.join(train_dir, "images")
    img_valid_dir_dir = os.path.join(valid_dir, "images")

    classes = [lbl.split("-", 1)[1] for lbl in os.listdir(dataset_initial_dir)]

    # Copy images to the training directory
    for train_img in train_paths:
        copy_img(img_train_dir, train_img[0], dataset_initial_dir, task=task)
        if task == "detect":
            full_anno_path = os.path.join("datasets/Annotation", os.path.splitext(train_img[0])[0])
            convert_annotation(full_anno_path, f"{train_dir}/labels/", classes)

    # Copy images to the validation directory
    for valid_img in valid_paths:
        copy_img(img_valid_dir_dir, valid_img[0], dataset_initial_dir, task)
        if task == "detect":
            full_anno_path = os.path.join("datasets/Annotation", os.path.splitext(valid_img[0])[0])
            convert_annotation(full_anno_path, f"{valid_dir}/labels/", classes)

    print("Dataset split into training and validation sets.")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(valid_paths)}")

    if test is not None:
        test_dir = "datasets/yolov8_imgs/test"
        img_test_dir = os.path.join(test_dir, "images")
        for test_img in test:
            copy_img(img_test_dir, test_img[0], dataset_initial_dir, task)
            if task == "detect":
                full_anno_path = os.path.join("datasets/Annotation", os.path.splitext(test_img[0])[0])
                convert_annotation(full_anno_path, f"{test_dir}/labels/", classes)
        print(f"Test samples: {len(test)}")
    else:
        test_dir = None

    create_yaml(train_dir, valid_dir, classes, test_dir, name=name)



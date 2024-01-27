# DogBreed
### General
This is a project tries to classify dog images into a 
predefined number of breeds. First we inspect that data consist of 120 classes of different Dog Breeds.

To ensure a balanced distribution between labels in the dataset, we plot a graph showing the samples.
We see that the 120 classes consist of approximately 80-100 instances per category so we conclude that
we have a general balanced dataset.
![img_1.png](img_1.png)

Data augmentation is applied by default during training using YOLOv8 ([config](https://docs.ultralytics.com/usage/cfg/#augmentation)).
![img_2.png](img_2.png)

One challenge with this dataset is that images exhibit different styles in terms of background, and they are of varying shapes. The strategy we will follow to address this issue is as follows:
Another challenge is that dog breeds are very similar with each other and that we dont have many 
samples considering the 120 number of classes.

### Approaches
#### 1) Easy Solution 
This approach is based on addressing the problem as a classification task, ignoring annotations. 
This straightforward approach treats the task as a vanilla classification problem, 
which is the fastest and easiest solution. 
However, it may yield undesired results when dealing with images containing multiple 
annotations or diverse backgrounds. <br /> 
Example:<br /> 
![img.png](img.png)
#### 2) Improved Solution
Another sophisticated solution involves training YOLOv8 as an 
object detector using the initial annotations (boxes of dogs) with multiple breeds. 
The model outputs boxes with labels, but we focus on the class (breed) rather than 
the box coordinates. The key metric is loss_cls, 
measuring the correctness of the classification, not loss_bbox. 
#### 3) Two step Solution 
This approach comprises two models. 
The first detects the dog in the image, treating all labels with categories as a single 'Dog.' 
The YOLOv8 object detector locates the dog, providing a bounding box. 
The second model is a classifier directly applied to the cropped object, aiming to classify the breed.
By doing this we help the classifier to decide using as input only the dog frame and not the dog with the 
background which might be confusing.
#### 4) Crop Classification Solution 
Another approach for this problem would be using annotations only in the training set.
As a first step would be to crop images and let valid and test set have the complete information. The task is a classification 
problem in this case.

### How to use this repository
If you want to utilize my repo without building an anaconda env you can simply build
a docker image (for example 'dogbreed' but you can also use another name) : <br /> 
`docker build -t dogbreed .`
After you build the image you can run: <br /> 
`docker run dogbreed`

### Setup-Parameters
The training flow can be found in main_train.py while the evaluation of the trained models 
can be found in main_results.py. In order to run those files you need to have the dataset unzipped
in your directory. In case you use the docker image this step is not required as the docker image includes
those data. 
First you need to pick which solution/s you want to use between ['SOLUTION_1', 'SOLUTION_2']. Then 
the following parameters can be tried:
model size between ['n','s','m','l','x'],
For example for 'n' the [yolov8n.pt](https://docs.ultralytics.com/models/yolov8/) will be used for training(n = nano).
Due to limited resources the models I trained were of sizes 's' and 'm'. 
The initial set consists of train and test set. Train set consists of 12000 samples while test set 
valid_size=0.2,
random_state=42,
yolo_sizes=yolo_sizes,
model_kwargs={'epochs': 100, 'imgsz': 120,
            'batch': 32, 'patience': 10},
save=True

The dataset will be split into training and validation given a validation ratio (0-1).
We utilize a random state of 42. 
The yolov8_imgs will be built before training which follows the requested dataset format
for yolov8. This folder will automatically removed after training.  <br /> 
The structure is the following: <br /> 
![img_3.png](img_3.png)


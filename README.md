# Dog Breed Classification Project
### General
This project aims to classify dog images into 120 predefined breeds. 
The dataset appears balanced, with each of the 120 classes having approximately 80-100 instances.
To ensure a balanced distribution between labels in the dataset, we plot a graph showing the samples label 
occurence.
We see that the 120 classes consist of approximately 80-100 instances per category so we conclude that
we have a general balanced dataset.
![img_1.png](img_1.png#center)

Data augmentation, using YOLOv8, is applied during training by default to improve model performance.
(see [config](https://docs.ultralytics.com/usage/cfg/#augmentation)).
![img_2.png](img_2.png)

### Challenges
1. The dataset exhibits different styles in terms of background, and images have varying shapes.
2. Some dog breeds are have similar appearance, and the dataset lacks sufficient samples for the 120 classes.

### Approaches
#### 1) Easy Solution 
This approach treats the task as a classification problem, ignoring box annotations. 
This straightforward approach is fast but may yield undesired results when dealing with images containing multiple 
annotations or diverse backgrounds. <br /> 
Example:<br /> 
![img.png](img.png#center)
#### 2) Improved Solution
A more sophisticated solution involves training YOLOv8 as an 
object detector using the initial annotations (boxes of dogs) with multiple breeds. 
The model outputs boxes with labels, and we focus on the class (breed) rather than 
the box coordinates. The key metric is loss_cls, measuring the correctness of the classification, not loss_bbox. 
#### 3) Two step Solution (not implemented)
This approach comprises two models. 
The first detects the dog in the image, treating all labels with categories as a single 'Dog.' 
The YOLOv8 object detector locates the dog, providing a bounding box. 
The second model is a classifier specifically applied to the cropped object. Its goal is to classify the dog breed.
This two-step process is designed to assist the classifier in making decisions based solely on the dog frame, 
excluding background elements that might introduce confusion to the model.
#### 4) Crop Classification Solution (not implemented) 
The initial step involves cropping images based on the provided annotations. 
This steps does not affect validation and test set. 
Utilizing the cropped images as the training set we classify the breeds. 
One drawback of this method is that the test set may significantly differ from the training set, 
particularly in images with divergent backgrounds.

### How to use this repository
If you want to utilize this repository without building an Anaconda environment, you can simply build a Docker image. 
For example, let's name it 'dogbreed': <br /> 
`docker build -t dogbreed .`<br /> 
After building the image you can run which will open a bash shell: <br /> 
`docker run dogbreed`

### Setup-Parameters
To initiate the training process, refer to `main_train.py`, and for evaluating the trained models, check `main_results.py`. 
Ensure the dataset is unzipped in your local directory before running these files. 
If you're using the Docker image, this step is not required as the docker image includes
those data. <br />
Before running the training and evaluation scripts, you may choose the solution/s you want to implement. 
Options include ['SOLUTION_1', 'SOLUTION_2'].
#### Model Size
This repo also allows you to experiment with different model sizes. 
Choose from ['n', 's', 'm', 'l', 'x']. 
For instance, selecting 'n' will use the [yolov8n.pt](https://docs.ultralytics.com/models/yolov8/) for training (where 'n' stands for nano). 
Due to limited resources, models were trained with size 's'.

#### Dataset Information
The initial set consists of train and test set. 
Train set consists of 12000 samples. First step of Trainer object is to split the training data into train-valid in a stratified manner. 
Since examples are not many with respect to number of classes we need to guarantee that the number of samples 
for each class is balanced.  <br /> 
For the splitting we utilize two parameters valid_size and random_state. 
The random_state is the seed used by the random number generator which we set to 42, Valid_size parameter 
should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the validation split.
For all the experiments we set this parameter to 0.2 which is a common standard. 

#### Model Training Parameters
We also have other adjustable parameters such as epochs, batch size and patience. 
These are set as following:  <br /> 
`model_kwargs={'epochs': 100, 
               'batch': 32, 
               'patience': 10}` <br /> 
Note that by using model_kwargs any training configuration parameters found [here](https://docs.ultralytics.com/usage/cfg/) can passed into the training

The yolov8_imgs folder will be built before training by the Trainer following the requested dataset format
for yolov8. This folder will automatically removed after training.
The structure is the following: <br /> 
![img_3.png](img_3.png)

The results are not good since the model is not able to identify classes correctly.
This could be due to many factors such as the number of classes which is high 
with the number of samples being low. On the other hand we have build a repo with complete
functionality for utilizing different parameters and setups. 


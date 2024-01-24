# DogBreed


This is a project tries to classify dog images into a 
predefined number of breeds. First we have no information considering the
data 

A problem with this dataset is that images present different style in terms of
background. Moreover, images are of different shape. The strategy we will follow is the 
following. 
### 1) Easy Solution
Address this problem as a simple classification problem ignoring annotations.
This solution is not suggested in general we are just basically trying to have a 
baseline / vanilla solution.
The problem with this approach is that we have images with multiple annotations for example the following
image contains two Rhodesian ridgeback dogs.
![img.png](img.png)
### 2) Solution
We will crop images using the annotations keeping only the dog boxes. 
Images will be resized a RGB image (shape, shape, 3). 
Shape is a hyperparameter which we will set as 224.


# DogBreed


This is a project tries to classify dog images into a 
predefined number of breeds. First we have no information considering the
data. We inspect that data consist of 120 classes of Dog Breeds.

To ensure a balanced distribution between labels in the dataset, we plotted a graph showing 120 classes with approximately 80-100 instances per category.
![img_1.png](img_1.png)

One challenge with this dataset is that images exhibit different styles in terms of background, and they are of varying shapes. The strategy we will follow to address this issue is as follows:

### 1) Easy Solution
Address this problem as a simple classification problem ignoring annotations.
This solution is not suggested in general we are just basically trying to have a 
baseline / vanilla solution.
The problem with this approach is that we have images with multiple annotations for example the following
image contains two Rhodesian ridgeback dogs.
![img.png](img.png)
### 2) Improved Solution
To overcome the challenges, we will crop images using the provided annotations, keeping only the dog boxes. The images will be resized to an RGB format with dimensions (shape, shape, 3). Here, 'shape' is a hyperparameter that we will set to 224.


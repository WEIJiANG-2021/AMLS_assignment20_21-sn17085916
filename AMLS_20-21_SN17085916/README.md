### Introduction

In this experiment, we use CNN network to conduct all the classification tasks. All tasks are implemented with `PyTorch`. 

### Training Process

1. initialize the convolution kernel W and other parameters b
2. The input picture is forward propagated, through the convolution, Relu and pooling layers in turn, and finally through the softmax output of the fully connected layer for the picture.
3. Calculate loss, use cross entropy loss for classification problem.
4. Gradient back propagation, update the weight value of the convolution kernel, usually using stochastic gradient descent.
5. Repeat steps 3 and 4 until the result converges

### File 

​	datadet.py    --Read data information to return image and feature tags
​    lenet.py      --CNN network
​    gender.py     --The main program for judging the gender of a character, which contains relevant parameters, training functions and test functions
​    emotion.py    --The main program for judging the expression of a character, which contains relevant parameters, training and testing functions
​    face_shape.py --The main program for judging the face shape of a character, which contains relevant parameters, training and testing functions  

 eye_color.py  --The main program for judging the pupil color of a character, which contains relevant parameters, training and testing functions.

### Required packages

​	os
​    numpy
​    pandas
​    torch
​    torch.nn
​    collections.namedtuple
​    torch.utils.data.DataLoader
​    torchvision.transforms
​    torch.optim

### Results

Accuracy on test dataset

| gender | emotion | face_shape | eye_color |
| ------ | ------- | ---------- | --------- |
| 0.94   | 0.85    | 0.84       | 0.83      |
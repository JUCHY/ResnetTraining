# ResnetTraining
Program using PyTorch that trains the Convolutional Neural network using a set of images

#images have not been added yet as they are too big
Assign3.py trains a resnet using a set of training images, the images in imagenet12_train have been divided into two sets, training and validation. After the training is done, the program then checks the accuracy using a set of images in imagenet_12_val
Assign4.py, tests for and plots the loss and accuracy of the predictions for multiple different sets of hyperparameters. It firsts checks the plot and accuracy for SGD(stochastic gradient descent) with learning 0.1, 0.01, and 0.001 and plots them on the same graph to compare them.
We then check out a different optimization algorithm, RMSprops, and try it out with three different sets of hyper parameters, the first set is the default settings that come with RMSprop, and the next two sets involved some randomly changed variables on my part. Their loss over time and accuracy over time(epoch) are also plotted.

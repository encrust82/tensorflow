# The Complete Guide to TensorFlow for Deep Learning with Python

Course notes from Udemy course The Complete Guide to TensorFlow for Deep Learning with Python.

## Installation and Setup

Anaconda
https://www.anaconda.com/

## What is Machine Learning

_Supervised Learning_ uses labeled data to predict a label given some features.  If the label is continuous, its called a regression problem, if its categorical it is a classification problem.  Supervised learning has the model train on historical data that is already labeled.  Once trained, it can then be used on new data.

_Classification Problem:_ Given the features height and weight, predict their gender (label).

_Regression Problem:_ Given the feature square footage and rooms, predict the selling price (label).

_Unsupervised Learning_ occurs when there are no labels.  Instead, a data scientist uses the results of _clustering_ to determine the results.  Clustering will not tell you what the labels should be.  Only that the points in each cluster are similar to each other based off the features.

_Reinforcement Learning_ works through trial and error which actions yield the greatest rewards.  Some tasks are teaching a program to play a video game, drive a car, etc.

There are components to reinforcement learning:

1) Agent-Learning/Decision Maker
2) Environment - What agent interacts with
3) Actions - What the agent can do

In the process of reinforcement learning, the agent chooses actions that maximize some specified reward metric over a given amount of time.  It learns the best policy with the environment and responding with the best actions.

The process for machine learning follows:

1) Data Acquisition: Obtaining data that will be used (i.e. ImageNet, IMNIST, etc.)
2) Data Cleaning: Processing the data into a format that you can use in your network
3) Split the data into training and test data (optionally also a holdout set)
4) Train: Train the model utilizing part of the data
5) Evaluate: Evaluate the model and produce metrics
6) Deploy Model: Utilize your model on new data.

Metrics for evaluating a machine learning model:

_Supervised Learning (Classification)_
Accuracy: Correctly Classified / Total Samples
Recall: 
Precision: 

_Supervised Learning (Regression)_
MAE: 
MSE: 
RMSE: 
On average, how far off are you form the correct continuous value

_Unsupervised Learning_
Cluster Homogeneity
Rand Index

_Reinforcement Learning_
How well the agent performs the task at hand.

## Neural Networks Introduction

A _perceptron_ is the computerized version of a biological neuron.  It is made up of inputs, weights, activation function, bias, and output (s).

The _bias_ term is used to help the issue of where inputs are 0.

```math
x0w0 + x1w1 + b = y
```

_Neural Networks_ are networks of perceptrons that are composed of different layers (input, hidden, and output).  More complex networks can have different layers that will be explained later.

_Activation Functions_ are used to determine if the neuron is "turned on" based on the weighted sum of the inputs.  _ReLu_ is one activation that generally has the best performance in a network.  Others include tanh, sigmod, and step.

_Cost Functions_ are used to determine how far off the value of a neuron is from the expected value.  The most common cost function used is the _Cross Entropy_ cost function due to the faster learning that is achieved.  _Quadratic Cost_ is another cost function that can be used.

_Gradient Descent_ is an optimization algorithm for finding the minimum of a function.

_Backpropagation_ is used to calculate the error contribution of each neuron after a batch of data is processed.  It relies heavily on the chain rule to go back through the network and calculate the errors.

A _Dense_ network is one where each neuron in a layer is connected to all of the neurons in the next layer.

[Tensorflow Playground](https://playground.tensorflow.org)

## TensorFlow Basics

[TensorFlow](https://www.tensorflow.org/) is an open source library used to develop and train ML models.

A _tensor_ is a fancy term for a n-dimensional array.

TensorFlow can be used to perform operations such as addition and multiplication.  The more interesting use for it is developing neural networks and performing machine learning.

A _graph_ is a sequence of inputs, operations, and outputs that produce some result.  Neural networks are graphs based on biological neurons.  

_Variables_ hold the values of weights and biases throughout a session, and must be initialized.

_Placeholders_ are initially empty and are used to feed in the actual training examples.  They must also be initialized.

The _Estimator API_ is a high level API that can be used to solve machine learning problems.  There are other APIs that are similar such as Keras and Layers.

In general,  the format to use the Estimator API follows:

1) Define a list of feature columns
2) Create the estimator model
3) Create a data input function
4) Call train, evaluate, and predict methods on the estimator object

**A good sign that you are not overfitting is that your training data loss is similar to your eval loss**

## Convolutional Neural Networks (CNNs)

_Xavier (Glorot) Initialization:_  Uniform/normal which draws weights from a distribution with zero mean and a specific variance.

_Learning Rate:_ Defines the step size during gradient descent.

_Batch Size:_ Batches allow us to use stochastic gradient descent.  Smaller = less representative of data and larger = longer training time.

_Second-Order Behaviour_ of the gradient decent allows us to adjust our learning rate based off the rate of descent.  i.e. AdaGrad, RMSProp, Adam.

_Unstable / Vanishing Gradients_: As you increase the number of layers in a network, the layers towards the input will be affected less by the error calculation occurring at the output as you go backwards through the network.  Initialization and normalization help to mitigate these issues.

_Overfitting:_ occurs when you train a model exactly to a dataset.  (Error is very low on training data, but high on test data).  You can use _L1/L2 Regularization_ to add a penalty for larger weights in the model (mitigating overfitting).  

_Underfitting:_ occurs when you don't train a model enough to get the results you require on a new dataset. (High error on training and test data).

_Dropout:_ remove random neurons during training.  This also helps mitigate overfitting.

_Expanding Data:_ artifically expands data by adding noise, tilting images, adding low white noise to sound data, etc.

A class data set in Deep Learning is the _MNIST data set._  This is a dataset made up of drawn characters, and can be used to recognize characters from text.  

TensorFlow has easy access to the data set with 55,000 training images, 10,000 test images, and 5,000 validation images.

A single digit image can be represented as an array (values between 0 and 1, grayscale), specifically 28x28 pixels.  We can flatten the array into a 1-D vector of 784 numbers.  However, flattening an array causes us to lose the relationship of a pixel to its neighboring pixels.

**We can think of the entire group of the 55,000 images as a tensor (an n-dimensional array).**

For the labels, we will use _One Hot Encoding_.  One Hot Encoding changes string labels into a single array.  The label is represented based off the index position of the label array (It will be a 1 at the index location, and 0 everywhere else).  For the MNIST training set, it will become a 2-d array (10, 55000).

A _Softmax Regression_ returns a list of values between 0 and 1 that add up to 1.  This means we can use this as a list of probabilities (that add up to 1).  _Softmax_ is an activation function that we will use as part of the Softmax Regression algorithm.

_Tensors_ are n-dimensional arrays that we build up to:

1) Scaler - 3
2) Vector - [3,4,5]
3) Matrix - [[3,4], [5,6], [7,8]]
4) Tensor - [[ [1,2], [3,4]], [[5,6], [7,8]]]

Tensors make it really convenient to feed in sets of images into a model (I,H,W,C):

I: Images
H: Height of image in pixels
W: Width of image in pixels
C: Color Channels -- 1-Grayscale, 3-RGB

Dense Layer: Every unit is connected to every unit in the next layer.

Convolutional Layer: Each unit is connected to a small number of nearby units in the next layer.

Convolutional Neural Networks (CNNs):  Help reduce the number of parameters.  Additionally, pixels that are nearby each other are much more related than pixels that are far apart.  Each CNN layer looks at an increasingly larger part of the image.  Having units only connected to nearby units also aids in invariance.  CNNs also help with regularization, limiting the search of weights ot the size of the convolution.

_Filters:_ Are represented by 2-d grids.  Filters detect certain features in an image (i.e. edges).

_Pooling Layers:_ Subsample the input image, which reduces the memory use and computer load as well as reducing the number of parameters.

A _kernel_ is a matrix of values that can be used to perform convolutions.

_Dropout_ can be thought of as a form of regularization to help prevent overfitting.  During training, units are randomly dropped, along with their connections.

Some famous CNNS include:

1) LeNet-5 by Yann LeCun
2) AlexNet by Alex Krizhevsky et al.
3) GoogLeNet by Szegedy at Google Research
4) ResNet by Kaiming He et al.

## Essential Machine Learning Q/A

1. What's the tradeoff between **bias** and **variance**?

_Bias is error due to erroneous or overly simplistic assumptions in the learning algorithm you're using.  This can lead to the model underfitting your data, making it hard for it to have high predictive accuracy and for you to generalize your knowledge from the training set to the test set._

_Variance is error due to too much complexity in the learning algorithm you are using.  This leads to the algorithm being highly sensitive to high degrees of variation in your training data, which can lead you model to overfit the data.  You'll be carrying too much noise from your training data for your model to very useful for your test data._

2. What is the difference between supervised and unsupervised machine learning?

_Supervised learning requires training labeled data.  Unsupervised learning does not require labeled data._

3. How is **KNN** different from **k-means clustering**?

*K-Nearest Neighbor (KNN) is a supervised classification algorithm, while k-means clustering is an unsupervised clustering algorithm.  Since KNN is a supervised classifiation algorithm it will need labeled data.  K-meanss clustering is unsupervised meaning it does not need labeled data.. and instead works off of clustering.*
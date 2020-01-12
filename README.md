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


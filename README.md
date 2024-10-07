# NeuralNetworkFromScratch

## Table of Contents
1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Model Architecture](#3-model-architecture)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
6. [Evaluation](#6-evaluation)
7. [License](#7-license)

## 1. Introduction

"NeuralNetworkFromScratch" is a project aimed at building a simple Multi-Layer Perceptron (MLP) from scratch without using deep learning frameworks like TensorFlow or PyTorch. The MLP is designed to classify handwritten digits from the MNIST dataset, which consists of 28x28 pixel grayscale images. This project implements a neural network with basic components such as fully connected layers, activation functions (Sigmoid), and a softmax output.

The project is developed to run on Kaggle and includes basic data preprocessing, training, and testing functionalities.

## 2. Dataset

The datasets useds for training and testing the model are the MNIST and CIFAR-10 datasets. You can download them from Kaggle via the following link:

- [MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [CIFAR-10 Dataset on Kaggle](https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders)

### Dataset Instructions
After downloading the dataset, update the `dataset` variable in the code with the path where the dataset is stored:
```python
path = '/kaggle/input/mnist-dataset'
or
path = '/kaggle/input/cifar10-pngs-in-folders/cifar10'
```

## 3. Model Architecture

The neural network used in this project is a simple Multi-Layer Perceptron (MLP) with the following structure:

- Input Layer: 28x28 pixels (784 features)
- Hidden Layer: 200 neurons, followed by a Sigmoid activation function
- Output Layer: 10 neurons (corresponding to the 10 classes of digits), followed by a Softmax function

### Model Code
```python
class FCN:
    def __init__(self, activation=sigmoid, lr = 1e-4):
        self.linear1 = Linear(28*28, 200)
        self.linear2 = Linear(200, 10)
        self.activation = activation
        self.softmax = softmax
        self.criterion = NLLLoss()
        self.lr = lr
        
    def forward(self, x):
        self.hidden = self.linear1.forward(x)
        self.activated = self.activation(self.hidden)
        self.output = self.linear2.forward(self.activated)
        self.probabilities = self.softmax((self.output))
        return self.probabilities
```
or
```python
class ConvNetwork(NeuralNetwork):
    def __init__(self, activation = ReLU(), lr = 1e-4):
        super().__init__(activation, lr)
        
        self.conv1 = Conv2d(3, 32, kernel = (3, 3), padding = 1)
        
        self.dropout = Dropout(p=0.5)
        
        self.pooling = MaxPool2d()
        
        self.conv2 = Conv2d(32, 64, kernel = (3, 3), padding = 1)
        
        #self.conv3 = Conv2d(64, 64, kernel = (3, 3), padding = 2)
        
        self.fc1 = Linear(4096, 500)
        
        self.fc2 = Linear(500, 10)
        
        self.softmax = Softmax()
        
        self.criterion = NLLLoss()
        
    def forward(self, x):
        out = self.conv1(x)
        
        out = self.activation(self.dropout(out))
        
        out = self.pooling(out)
        
        out = self.conv2(out)
        
        out = self.activation(self.dropout(out))
        
        out = self.pooling(out)
        
        #out = self.conv3(out)
        
        #out = self.activation(self.dropout(out))

        out = out.reshape(out.shape[0], -1)

        out = self.fc1(out)
        
        out = self.fc2(out)
        
        return self.softmax(out)
```

### Training and Testing
The model is trained for 5 epochs, and the results are evaluated using accuracy. The training loop handles loss calculation, backpropagation, and weight updates.

## 4. Installation
Prerequisites
- Python 3.7 or later
- NumPy
- Matplotlib

### Setup Instructions
1. Download or clone the repository to your local machine or Kaggle environment.
2. Download the MNIST dataset from Kaggle.
3. Download the CIFAR-10 dataset from Kaggle
4. Place the dataset files in the directory and update the `dataset` path in the script.

## 5. Usage
1. Upload the script `NeuralNetworkFromScratc` and the dataset.
2. Update the `dataset` path in the code to match the location of the dataset.
3. Run the script to preprocess the data, train the model, and evaluate the results.

### Running The Model
```python
train_dl = DataLoader(mnist_train_data, batch_size = 32, shuffle = True)
test_dl = DataLoader(mnist_test_data, batch_size = 32)

model = MLP()

accuracy, loss = model(mnist_train_dl, num_epochs = 5)

result = model.test(mnist_test_dl)
```
or
```python
model = ConvNetwork()
history = model(cifar_train_dl, cifar_valid_dl, num_epochs = 5)
test_accuracy = model.test(cifar_test_dl)
```

## 6. Evaluation
The model performance is evaluated using accuracy on the test set. Additionally, the loss is monitored during training. Key metrics:

- Accuracy: Shows the percentage of correctly classified digits.
- Loss: Cross-entropy loss is computed using the `NLLLoss` function.

## 7. License
This project is licensed under the [MIT License](LICENSE).

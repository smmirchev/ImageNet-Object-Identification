# ImageNet Object Identification 
The project software creates a machine learning model with the help of the Python neural-network libraries - Keras and Tensorflow on the backend.The model was trained on an ImageNet dataset, consisting of approximately 1.3 million images, divided into 1000 categories. The model achieved 84% accuracy on the testing data and is able to accurately predict objects in images, if these objects are within the 1000 categories. To allow for a better presentation and more user-friendly interface the project was build using Python Flask and deployed to the web with the help of JavaScript

<img src="https://github.com/smmirchev/ImageNet-Object-Identification/blob/master/images/Interface.png">

## Getting Started
Steps to start the ImageNet project:
1. Download the ImageNet Object Identification Challenge repository
2. Place all the files in one folder
3. In this folder create an empty folder called 'static'
4. Inside the static folder create a folder called 'uploaded'
5. Train the model by starting the CNN.py file ** check "Installing" below**
6. Start the Interface.py file as a flask app
7. Open the flask website link
8. Click on the 'Choose Image' button and select an image
9. Click on the 'Get Predictions' button to see the results 


### Prerequisites
The following python libraries must be installed to run the project.
* Python 3.6
* flask 1.1.1
* keras 2.3.1
* numpy 1.18.1
* tensorflow 2.1.0
* setuptools 45.2.0
* matplotlib 3.2.1
* pandas 1.0.3

### Installing
The steps described in "Getting Started" are explaining how to get predictions for images. To train the model a dataset is required. The dataset used by this model is not uploaded as it is a single .tar file with size of 540GB. The dataset can be downloaded from here "https://www.kaggle.com/c/imagenet-object-localization-challenge". The model can be trained on any image dataset if its performance is to be tested. However, the CNN is quite complex and it may be too extreme for simple datasets, it is also quite computationally expensive.

## Author
* Stefan Mirchev

## References
* [ImageNet Dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge)

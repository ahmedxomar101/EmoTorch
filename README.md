# Facebook-AI-Hackathon2020
## Project **EmoTorch** is built for the Facebook Hackathon Contest in the Artificial Intelligence Track using the mentioned tool PyTorch.

![PyTorch](PyTorch.jpg)

**Contributors**

- [Ahmed Hamido](https://github.com/AhMedDxHaMiDo) - Egypt
- [Nathan Curtis](https://github.com/njcurtis3) - USA
- [Yashika Sharma](https://github.com/Yashika51) - India

**Face Emotion Recognition** project aims at predicting the emotions of the user based on his facial image.

The motivation for this project came from the recommendation systems.
Often we see products recommended based on the user's search history, watch history , purchase history etc. These sort of systems depend on the past history and records. We with this project want to use real-time recommendation system. The system will use our model to predict the emotion of the user based on his facial expressions and will recommend products to suit the mood of the user.


# Table of Content
1. [Dataset](#dataset)
   1. [Emotions](#emotions)
   2. [Data Representation](#data-representation)
2. [Used Libraries](#used-libraries)
3. [Network Architecture](#network-architecture)
4. [Model](#model)
5. [Hyperparameters](#hyperparameters)
6. [Performance](#performance)
7. [Further Additions](#further-additions)
8. [What is Next](#what-is-next)
9. [References](#references)
10. [License](#license)

## Dataset
The model is built entirely on the publicly available datasets. Since, the field **Facial Emotion Recognition(FER)** is not much developed yet we had very limited available datasets. Based on our research, we chose the FER dataset. 

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories.

### Emotions
| Class | Label |
| --- | --- |
| 0 | angry |
| 1 | disgust |
| 2 | fear |
| 3 | happy |
| 4 | neutral |
| 5 | sad |
| 6 | surprise |

The dataset is downloaded from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/overview). 

The reason behind choosing this dataset among others-
- It has images categorized in one of the seven emotions.
- Is publicly available
- The length of the dataset is suitble for our task with 
 - Training set:  28,709 examples.
 - Test set: 3,589 examples.
 -Validation set: 3,589 examples.

 ### Data Representation
The file train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and our task is to predict the emotion column.

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project.


## Used Libraries
- [PyTorch](https://pytorch.org/)- This is the main library used in this project. The model is entirely build on PyTorch including all the image transformation, transfer learning, training, testing.

- [NumPy](http://numpy.org/docs)- Used for pixels manipulation

- [Matplotlib](http://matplotlib.org/)- To plot images and loss plots.

- [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html)- Used for creating the dashboard for accuracy and loss plots


## Network Architecture
The motivation of using transfer learning for out task came after we implemented a Deep Neural Network from Scratch. The model build from scratch gives accuracy only around 18%-20%. We can boost the accuracy with the help of transfer learning.

PyTorch's models module has variety of pretrained networks which can be easily downloaded.

In our project, we tried multiple networks before settling to VGG19.

Initially we used VGG16 which gave us accuracy below 40% followed by ResNet50 with 41% and DenseNet101 with 42.5%. VGG19 yields an accuracy of 46% which is better than all other pretrained model. 

Therefore, we decided to choose VGG19 for the implemetation. 


## Model
The pretrained models are trained on the ImageNet Dataset classifying 1000 classes. Our task is to classify the face expression in one of the 7 classes, so we modified the classification layer and replaced it with our Network.
The final model is a combination of transfer learning and custom trained classifier.


## Hyperparameters
The training was done with images in a batch of 64. The learning rate was set nethier too large nor too small that to 0.0001.

To avoid overfitting we used dropout layers in between hidden layers followed by ReLU activation function.
The number of epochs are 25 and the hidden layers are 1024 because the model did not improve beyond that.

The optimzer chosen is Adam.


## Performance
The model is completely built on the public free available dataset in contrast to the commercial FER projects that use large datasets with high resolution quality. However, the model is capable of being trained on any dataset and predicting the accurate emotions. 
Currently the accuracy is 46% which can be improved with diverse datasets.


## Further Additions
The distribution of samples per category in the FER dataset is not balanced. 
The category disguist is least represented with only 547 samples whereas the category happiness is most represented with 8989 samples.

Future Scope lies in augmentation. Multiple balancing techniques can be used to present equal number of apparations per category which will result into higher accuracy.


## What is Next
The model is ready and predicts accurate emotions based on the image of a person's face. Our aim to build this project is to merge it with Facebook's product recommendation system.

The predicted labels will be sent to the recommendation system which will inturn predict the most likely products.

The front camera of a cell phone or laptop will capture the face of the person on consent while browsing Facebook Feed. Alternatively, a selfie can be fed. This image is sent to our model which predicts the emotion of the user. This emotion is sent to the recommendation system which predicts the products based on those emotions.

This will help in better product recommendation which in turns will increase the probability of customer buying or clicking on a product.


## References
> Thanks to [Mahdi Ashraf](https://github.com/MahdiAshraf) for reviewing the data setup

- [FER Dataset](https://datarepository.wolframcloud.com/resources/FER-2013)

- [Article reference for Tensorboard](https://medium.com/looka-engineering/how-to-use-tensorboard-with-pytorch-in-google-colab-1f76a938bc34)

- [Stack Overflow Answer on using tensorboard in colab](https://stackoverflow.com/a/48468512/1514728)

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)


## License
[**MIT License**](https://choosealicense.com/licenses/mit/)

***Inspired by 2020 Facebook AI Hackathon Contest***.

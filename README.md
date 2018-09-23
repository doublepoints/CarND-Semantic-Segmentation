# Semantic Segmentation
### Introduction
In this project, I have tried to label the pixels of a road in images using a Fully Convolutional Network (FCN), which is based on the VGG-16 image classifier architecture. The dataset is from the KITTI dataset.

### Setup
***
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
***
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Approach
***

#####Architecture

A pre-trained VGG-16 network was converted to a fully convolutional network by converting the final fully connected layer to a 1x1 convolution and setting the depth equal to the number of desired classes. Performance is improved through the use of skip connections, performing 1x1 convolutions on previous VGG layers (in this case, layers 3 and 4) and adding them element-wise to upsampled (through transposed convolution) lower-level layers (i.e. the 1x1-convolved layer 7 is upsampled before being added to the 1x1-convolved layer 4). Each convolution and transpose convolution layer includes a kernel initializer and regularizer.

#####Optimizer
The loss function for the network is cross-entropy, and an Adam optimizer is used.

#####Traning
The parameters used for traing are shown as:
	- keep_prob: 0.5
	- learning_rate: 0.0009
	- epochs: 50 
	- batch_size: 1
	- weights_regularized_l2 = 0.001

### Result
***

The loss average durning training is showed as below:
![loss_epoch](https://github.com/doublepoints/CarND-Semantic-Segmentation/blob/master/Figure_1.png) 


After the first epoch, loss is 0.25 and  at epoch 50, the error is 0.011.

#####Samples

Thera are some sample images from the output of FCN shown as following, and the segmentation is marked in green.

![](https://github.com/doublepoints/CarND-Semantic-Segmentation/blob/master/uu_000085.png) 
![](https://github.com/doublepoints/CarND-Semantic-Segmentation/blob/master/umm_000065.png) 

 
### Conclusion
In this project, I have realized the segementation on lane from a image. However, FCN which is proposed in 2014 is not suitable for the high precision need, the following work should be focused on the high performance models, for example Mask-rcnn, deeplab etc. 

# Texture Detection 

## By Matthew Mepstead and Colin Shaw


## ABSTRACT

The purpose of this project was the description of images based
upon the textures they displayed. By observing the qualities of the
textures, it is possible to give proper language descriptors to images
in order to accurately describe them. Utilizing a data set called, the
Describable Textures Dataset (DTD) it is possible to build up a
baseline in order to design a neural network that can classify images
using the 47 different descriptors chosen to represent the varying
qualities of textures. Due to hardware limitations though the only
three image classifications of the 47 that could be run were: banded,
dotted, and zigzagged. This project looks at the creation of such a
neural network based in TensorFlow and looks at the applications
of such technology. Through the course of the paper the creation
process for the neural network will be laid out, the issues that were
encountered during the production will be explained, it will explore
the accuracy and usefulness of the algorithm.

## 1 Describable Textures Dataset

The Describable Textures Dataset or DTD as it will be commonly
called through the course of this paper, is a set of images each
classified with a real-language descriptor. These descriptors were
established in the paper that this project was based on. Based on
work of psychologists they determined which words were most
used to describe textures and then selected from these words a
selection of 47 which they determined could collectively be used
to describe almost any texture. They then went about selecting
120 images of each of the 47 different classifiers from sources
such as Google and Flickr. By compiling all these images
together, they created the DTD. Some of the images and their
classifications are shown below in order to give an idea of what
kinds of images are being used for the creation of the neural
network.

![figure 1](https://github.com/Colin-Shaw2/textureDetection/blob/master/figures/figure%201.png)
```
Figure 1: A selection from the “Bumpy” classifier in the DTD
```

![figure 2](https://github.com/Colin-Shaw2/textureDetection/blob/master/figures/figure%202.png)
```
Figure 2 : Second selection from the “Bumpy” classifier to
properly demonstrate the range of a single classifier
```

![figure 3](https://github.com/Colin-Shaw2/textureDetection/blob/master/figures/figure%203.png)

```
Figure 3 : Selection from the “Marbled” classifier
```

## 2 Pre-processing

The first step in the creation of a classifying neural network is to
pre-process the desired images so that they are in the correct format
for convolution. This pre-processing phase requires a couple of
different steps. First, it needs to be ensured that all images are the
same size, and then the actual images need to have their features
extracted in order to be used for the neural network. The other steps
of normalization and flattening were found to be simpler to do in
the neural network which will be discussed later.

Ensuring that all the images were the same size was critical for this
project. If the images were all different sizes it would make them
very difficult to classify as there would be different data even for
those within the same classification. Since resizing to a power of
two is the simplest, the options were between 256x256 and
512 x512. The advantage of smaller images would be that there
would be less overall pixel data and so the workload on the
computer running the neural network would be reduced. The
problem with this however is that the lack of data is a double-edged
sword. The more data you can pull from a single image the more
information a neural network will have to do its classifications. So,
in the end resizing all the images to be 512x512 was decided to be
the best method to ensure that the images maintained a specific size
throughout the process. Most of the images were within 100 or so
pixels of this size anyways so the loss of data from the resize was
considered minimal. This resizing was made quite simple with the
OpenCV resize method which allows for resizing of any image to
any specified size. By applying this to every image in the dataset it
is possible to either upscale or downscale the image in order to fit
the desired dimensions. Aspect ratio was not always preserved as
many images were not square to begin with however the error this
could cause in the neural network was considered negligible.

After all the images were set to be the same in size the next step of
the process was to determine the feature of the images that would
be sent to the neural network. In order to do this, it needed to be
decided which features would be used for the classification. The
choice of feature is critical when it comes to classification. If the
selected feature does not accurately reflect the properties of the
image that are being classified, then the final neural network will
have poor results. For example, if a histogram of intensities was
chosen to represent the texture of an image the neural network
would not function as desired as the intensity of the pixels has
nothing to do with the texture that the image represents. The key is
to choose a feature that is descriptive of the image while at the same
time not using the entire image. Using the entire image could lead
to issues as demonstrated in the above images from the “Bumpy”
classification. The images within a specific classifier are of such a
wide range that submitting the entire image to the neural network
would lead to a much lower accuracy than intended.

The feature that was chosen for this project was the edges in the
image. The selection of edge detection for the feature was due to
the fact textures in a 2D environment are based upon the edges that
make them up. Sharp edges represent textures with well defined
borders such as banded textures, rounded edges are representative
of something like a bubbly texture, a lack of edges is clearly more
of a smooth texture. Although there are potentially better features
to select to describe the texture, it was decided that edge detection
would be accurate for our purposes. The actual edge detection of
the image was made quite simple due to the OpenCV Canny Edge
Detector. By utilizing this built in method, it was quite simple to
start pulling the edges from the images in the DTD.

That isn’t to say that there weren’t issues with the process. The
biggest of these was determining the hysteresis threshold to use for
each image. Since not all of the images had well defined edges (as
mentioned above, since the edges represent the texture quite well
and not all textures are well defined), different upper and lower
bounds were needed for each image in order to gather the edges in
a way that accurately represented this image. The problem of
automatic thresholding is quite complicated. Essentially, another
feature of the image needs to be examined in order to determine
what the best threshold for it would be. There are many options for
this but in this paper utilizing the median of the pixel values for the
image was determined to be the most effective after some tests with
various features. The median works well as a representative of the
threshold because it is the exact middle value, meaning that it can
guaranteed to capture a good portion of the data if the upper and
lower bounds are based around the median value. One of the other
features that was tested was the average pixel value of the image.
However, the average turned out to be unreliable as it was too
greatly skewed by a small number of high pixel values in the image.
This skew led to images with high pixel intensities in some areas to
having far too high of a lower threshold resulting in an edge
detection that was entirely black. The opposite was true if the pixel
intensities were too low. It would result in a very low threshold
resulting int almost everything in the image being considered an
edge. The median value covers both cases by choosing the absolute
middle value, resulting in an accurate threshold. The two thresholds
were set at plus or minus 30% of value. This range was tested across
the various classifiers in the DTD and was determined to be quite
accurate at pin-pointing the edges across the majority. Using this
hysteresis threshold for the Canny Edge Detector finally allowed
for the proper extraction of the edges for classification.


## 3 TensorFlow Neural Network

To classify the images a convolutional neural network was used.
The images were loaded using an ImageGenerator and a
DirectoryIterator. These loaded images from a directory
while randomly applying rotations and flips to make the
training data more robust as well as changing the pixel data
range from 0-255 to 0-1.

![figure 4](https://github.com/Colin-Shaw2/textureDetection/blob/master/figures/figure%204.png)
```
Figure 4: Summary of the Network
```

For the network there was a convolution layer with input shape
(512,512,1) with a (5,5) kernel, a rectify linear activation function
and a filter size of 16. The next layer then had a max pooling layer.
Following that there was another convolution layer with 32 filters,
a rectify linear activation function and a (5,5) kernel followed by a
max pooling 2D layer. Next, these layers were followed by a final
convolution layer with 64 filters, a rectify linear activation function
and a (5,5) kernel and then followed by another max pooling 2D
layer. There was a flatten layer after those to change the network
into one dimension. Then a dense layer with a rectify linear
activation function was added. Finally, the last layer is a dense
layer with an output space equal to the number of texture types used
for training. This layer had a SoftMax activation function so that
the probability was returned for each possible label. The model
used Adam as the optimizer and tracked the accuracy metric. The
model used the categorical cross-entropy loss function. This loss
function was chosen since the targets were hot-one encoded,
meaning every target was tensor with a size equal to the number of
textures that the network was training for and filled with 0s except
for the index representing the correct texture which had a value of
1. ie:[0, 1, 0]

80% of our data was training data and 20% was for validation. It
was found that training for about 20 epoch would yield the highest
validation accuracy. After about 20 epochs overfitting began to
occur. The highest validation accuracy that was achieved was 79%

![figure 5](https://github.com/Colin-Shaw2/textureDetection/blob/master/figures/figure%205.png)
```
Figure 5: Model training over 50 epochs, Epoch 23 has the
highest validation accuracy
```
## 3 .1 Methods used to increase accuracy

The above neural network was the final product of experimentation
using multiple different networks and preprocessing techniques.
The above model was the method that yielded the highest validation
accuracy.  All models that were experimented on were Sequential. The
experiment was done by using a simply neural network with no
convolution layers and it was found that the model could never get
training accuracy over 50%.

![figure 6](https://github.com/Colin-Shaw2/textureDetection/blob/master/figures/figure%206.png)
```
Figure 6: Dense only neural network
```

![figure 7](https://github.com/Colin-Shaw2/textureDetection/blob/master/figures/figure%207.png)
```
Figure 7: Dense only neural network Accuracy [y axis only goes
up to 60%]
```

It was then moved to using networks with convolution layers and
max pooling. This greatly increased the accuracy but lead to
overfitting where the accuracy would hit 95% while the validation
accuracy was only 45%.

![figure 8](https://github.com/Colin-Shaw2/textureDetection/blob/master/figures/figure%208.png)
```
Figure 8: Over fitting of a convolution model, Accuracy is 99%
while validation accuracy is around 50%
```

![figure 9](https://github.com/Colin-Shaw2/textureDetection/blob/master/figures/figure%209.png)
```
Figure 9: Comparison of the model using robust (left) and non-
robust(right training images)
```

The kernel size of the convolution layers was varied but it was
found that the accuracy would always be 33% (exactly random
chance) when the kernel size was increased above 5.

To fix this issue training and validation data was augmented by
using the image generator class. This class would randomly apply
rotations, flips and zoom to images as they were loaded in to make
the training data more robust and reduce overfitting. After using
the robust images for training the model trained “slower” (loss
decreased by less each epoch) but validation and training accuracy
stayed much more similar, showing that overfitting had been
greatly reduced.
## 3 .2 Limitations due to hardware

While running this on the available laptops there were some serious
issues involving memory use and computation time. When training
the models Python would simply say Killed and end the process
with no other error message. This error was either due to a time out
or lack of RAM. Sometimes when training also the RAM on the
laptops would become full and the program would read off the disk
essential becoming so slow that the program had to be killed.

To deal with this problem three classes of the data were used
(banded, dotted and zigzagged) instead of all 47, although the code
should be extendable to handle all the classes. The code was also
moved to Google Colab which solved the issues of running out of
RAM and randomly crashing.

## 4 Applications

The classification of textures is quite a useful tool. It is especially
useful in the fields of material recognition. The descriptors that
are given to a specific texture can be related to specific material as
a next step in the classification process. For example, it can be
said that a metal surface is usually “smooth” and “shiny”. If an
image that is run through the neural network from this paper
returns these two descriptors, it can be used to supplement other
classifier and determine whether the material in the image is
metal. Material recognition as well as basic texture description
can be extended to use in real-world applications such as
autonomous vehicles, where the conditions of the road and the
properties of objects the car sees can be determined based upon
the textures that the camera picks up. The kind of texture
description explored in this paper is also quite useful for
determining the qualities of images that contain a multitude of
textures. Since the classifier is simply applying the most likely of
descriptors to the image an image composed of multiple textures
will be labelled according to every texture it is composed of.
These descriptors act as a useful feature when attempting to
describe or classify a complicated scene with many different
elements in it. The real-world is not made up of single texture
scenes, every scene contains a multitude of materials, each with
their own textures.

In fields such as aerial image processing, and just general image
classification, each image is made up of potentially dozens of
different textures. With a neural network like the one created for
this project, the various textures can be classified using the
descriptors and the set of descriptors for an image can be used as a
feature for further classification. For example, if one attempted to
classify which room of the house an image was taken in, a neural
network like this project’s could be quite helpful. Each room in a
house is comprised of its own set of textures. Bathrooms tend to
have very smooth textures, kitchens tend to have marbled and
wooden textures combined, and bedrooms have a mix of textures
from knitted, to wooden, to bumpy. By generating the set of
descriptors for an image with this DTD neural network it would
be simple to then classify each type of room based on that set.
Aerial image processing is also an interesting field that this
project has applications to. Its similar to the ideas described above
for general image classification, where each aerial image is
comprised of large numbers of different textures and it would be
simple to classify what kind of area was being looked down upon
based on the set of descriptors given to the image’s textures.
However, in terms of aerial photography it extends beyond
classification into observing changes in an environment over time.
For example, forest degradation could easily be observed using a
texture descriptor since the forest itself would be described as one
kind of texture whereas the unforested area would be described
very differently. By observing the change in the probabilities of
each descriptor over time it would be possible to note changes in
forestation.

## 5 Extensions

Provided more time the most effective extension to this project
would be running even more images from the classifications
through as well as refining the neural network to an even higher
accuracy. This extension would require much more powerful
computers than what is currently being used. The number of images
that could be run was severely limited due to hardware, so provided
the appropriate materials it would be an important improvement to
make. Another possible extension would be running images with
multiple textures through the neural network and classifying them
based upon the descriptors that were returned. This is a real-world
application touched upon in the applications section above and
would prove the applicability of the project in a real-world scenario.
It would first require the use of the first extension mentioned,
namely being able to run the entire DTD through the neural network
in order to improve the accuracy of the classifications. The higher
accuracy would be especially important in a cluttered image due to
the high number of textures that are blending together.

## 6 Conclusion

Over the course of this project a neural network was designed that
can describe the textures in an image based upon the 47 different
descriptors that were established in the DTD. After pre-processing
these images by running edge detection, they are sent to the neural
network for processing. They are run through both convolutional
and dense layers in order to create a proper classification. It was
shown that images could be classified with up to a 65% accuracy
with their overall descriptor. Given images with multiple textures
the neural network would produce the descriptors that are the most
accurate for the textures it finds. This technology has applications
in things such as material detection and real-world areas such as
autonomous vehicles, aerial image processing, as well as just
general image classification. Texture description is a critical field
of computer vision and neural networks that can describe the
textures that make up an image in a way that is understandable, not
only to computers but to humans as well, are vital to the continued
research and application of many different fields.


## REFERENCES

[1] Cimpoi, M., Maji, S., Kokkinos, I. et al. Deep Filter Banks for Texture
Recognition, Description, and Segmentation. Int J Comput Vis 118, 65 – 94
(2016). https://doi.org/10.1007/s11263- 015 - 0872 - 3
[2] Load images: TensorFlow Core. (n.d.). Retrieved April 10, 2020, from
https://www.tensorflow.org/tutorials/load_data/images?fbclid=IwAR0pUNRv0I
wYDLqo05PuTbB_DbvOpHDsfTlhhMU6zsbMb1pNWl7_3LY6rY 4



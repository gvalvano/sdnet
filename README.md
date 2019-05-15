# Convolutional Neural Networks for the Segmentation of Microcalcification in Mammography Imaging

**Tensorflow implementation of Segmentator and Detector neural networks**


For details refer to the paper:

> Valvano, Gabriele, et al. "Convolutional Neural Networks for the segmentation 
> of microcalcification in Mammography Imaging." Journal of Healthcare Engineering 2019 (2019).

----------------------------------
**Data:**

For our experiments, we used 283 mammography images with a resolution of 0.05 mm. Among these images, there are both natively 
digital mammograms and digitized images. Every image is associated to the corresponding manual segmentation mask realized by a 
breast imaging radiologist. We randomly chose 231 mammograms and the annotated labels to build the training set while 25 mammographic
images were used to validate intermediate results and compare different networks architectures. +e remaining 27 images were taken 
apart to build the test set and measure the final performances.

We contemplated 4 possible classes of patch:
 - Class C1: patches whose central pixel belongs to a microcalcification
 - Class C2: patches with MCs close to the center but with the central pixel not belonging to a calcification
 - Class C3: cases where a calcification resides inside the patch but is located peripherally, and the central pixel does not 
   belong to a MC
 - Class C4: cases where no MC is present inside the patch

# How to use it

1. Prepare your own dataset
2. Split the data in train, validation and test set folders
3. Run *train.py* to train the model.

You can monitor the training results using TensorBoard running the command:
```bash
tensorboard --logdir=results/graphs
```
in your bash under the project folder.

---------------------

For problems, bugs, etc. please contact me :)
Enjoy the code!

**Gabriele**
# SDNet

**Tensorflow implementation of SDNet**


For details refer to the paper:

> Chartsias, A., Joyce, T., Papanastasiou, G., Williams, M., Newby, D., Dharmakumar, R., & Tsaftaris, S. A. (2019). 
> *Factorised Representation Learning in Cardiac Image Analysis*. arXiv preprint arXiv:1903.09467.

----------------------------------
**Data:**

Automatic Cardiac Diagnostic Challenge 2017 database. In total there are images of 100 patients, for which manual
segmentations of the heart cavity, myocardium and right ventricle are provided.

Database at: [*acdc_challenge*](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).\
An atlas of the heart in each projection at: [*atlas*](http://tuttops.altervista.org/ecocardiografia_base.html).

# How to use it

1. Download the ACDC data set
2. Split the data in train, validation and test set folders (e.g. using *split_data.py*)
3. Run *prepare_dataset.py* to pre-process the data. By doing this, the image pre-processing will be offline 
and you will be able to train the neural network without such an additional CPU overload at training time 
(there are expensive operations such as interpolations). Data will be:
    - rescaled to the same resolution
    - the slices will be placed on the first axis
    - resized to desired dimension (i.e. 128x128)
    - masks one-hot encoded
4. Run *train.py* to train the model.

You can monitor the training results using TensorBoard running the command:
> tensorboard --logdir=results/graphs

in your bash under the project folder.

---------------------

For problems, bugs, etc. please contact me :)
Enjoy the code!

**Gabriele**
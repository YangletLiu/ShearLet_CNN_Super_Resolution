# ShearLet_CNN_Super_Resolution
We combine shearlet with CNN in the application of image super-resolution.


Tested Framework:
![Image text](https://raw.githubusercontent.com/hust512/ShearLet_CNN_Super_Resolution/master/ImagesInReadme/framework-1.png)

Currently Trainning Framework (new):
![Image text](https://raw.githubusercontent.com/hust512/ShearLet_CNN_Super_Resolution/master/ImagesInReadme/framework-2.png)


Usage:

Use generate_train.m to generate input data of the CNNs;

Use generate_test.m to generate label data of the CNNs;

Train the CNNs on Caffe (The network configuration and trainning details are writen in SRCNN_net.prototxt and SRCNN_solver.prototxt);

Use demo_SR_image.m to test the trained CNNs.

Dataset:

The network is trained on BSDS500 dataset and some drone images found on the internet
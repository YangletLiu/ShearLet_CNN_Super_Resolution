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


程序包说明：

ShearletCNN_train_v3.3：不再是分别训练9个DST系数，而是将9个系数堆叠为三维张量进行训练，即原本需要训练9个网络，而现在需要训练一个网络。

ShearletCNN_v3.5：不再是将9个系数堆叠为三维张量，而是将前8个高频分量作为一个张量进行训练，第9个低频分量再进行训练。

ShearletCNN_v3.6：先将coeffs用shearletSystem.RMS进行标准化，再作为input和label，其余继承自v3.3。

ShearletCNN_train_v3.3.1：不是将bic_HR图像的DST系数作为输入，而是将bic_HR图像本身作为输入，其余继承自v3.3。

ShearletCNN_v3.7：结合v3.5和v3.6，双路径的同时，分别用shearletSystem.RMS进行标准化。

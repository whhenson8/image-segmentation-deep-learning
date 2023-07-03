# image-segmentation-deep-learning
Based on the paper (to be published) - Automatic segmentation of lower limb muscles from MR images of post-menopausal women based on Deep Learning and data augmentation.

## Description
In this repository, three convolutional neural network architectures are implemented: the UNet, Attention UNet (att-unet), and Spatial Channel UNet (sc-unet). 
1) UNet - Follows directly the paper by Ronneberger et al. 2015 (arXiv:1505.04597).
2) Att-UNet - Follows directly the paper by Oktay et al. 2018 (arXiv:1804.03999).
3) SC-UNet - Novel neural network architecture that combines imaging data and spatial data associated with medical images.

## Table of Contents

![image](https://github.com/whhenson8/image-segmentation-deep-learning/assets/136348259/994c4a40-5950-4e0b-9e27-db4e77c7eab3)
***Figure**: Flowchart showcasing contents of the repository. Elements with curved edges represent folders, the square element holds the executable scripts.*

There are two implementations of the UNet and Attention-UNet, both in PyTorch. model_simple.py outlines each stage explicitly to allow beginners to follow through the stages. model.py outlines the archetectures in a sligtly more complex, less memory intensive way. The SC-UNet is outlined without a simple implementation.

Please note that due to ethical regulations, no sensitive medical data is stored in this repository. 


## Getting Started
The code is simple to use and is commented in the areas that may need adjusting to suit your needs. Follow the steps outlined below to train these networks for your specific task.

### Using the networks
  1) Get your data set up. These models are implemented to allow many images captured from one subject to be trained on and segmented.   Each image should be labelled with a specific code containging subject ID and the Image number (e.g. SubjectID_ImageNumber ->          'MC1001_1.png'). If required, download images from: https://doi.org/10.15131/shef.data.20440164, and masks from:                       https://doi.org/10.15131/shef.data.20440203 and assign subjects to each of train, validation, and test in the 'data' folder.
  2) Check the code. Firstly, check the locations of the folders on your local machine - these are highlighted in the code. Secondly, check that the number of out_channels (classes) matches the number of classes in your segmentation task (e.g. if only checking for one tissue, no. of classes = 1).
  3) Change  hyperparameters. Ammend hyperparamters outlined in train.py to fit your preferences. You are welcome to leave some as they are but epoch and checkpoints should be manually altered.
  4) Run train.py. This will train the model. When changing hyperparameters and loading from previous checkpoints, take care of the checkpoint number (LOAD_EPOCH).
  5) When complete, use test.py to load the trained model and apply to unseen training data.


### Required packages
  1) Torch ($ pip install torch)
  2) Torchvision ($ pip install torchvision)
  3) re ($ pip install regex)
  4) tqdm ($ pip install tqdm)
  5) Pillow ($ pip install Pillow)
  6) numpy ($ pip install numpy)
  7) albumentations ($ pip install albumentations)
  8) pydicom ($ pip install pydicom)
  9) glob ($ pip install glob2)

## Contributing
The code is tied to a publication and is therefore not open for contributions.

## Acknowledgements
I would like acknowledge EPSRC for funding the project, my supervisory team: Prof. Claudia Mazzà, Dr. Enrico Dall'Ara, and Dr. Xinshan Li.

## Contact
Email: whhenson8@gmail.com

# Troubleshooting:
TBA

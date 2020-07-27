# Todo

## Setup infrastructure
- sign up and use the credits (svenja sign up, Matheus setup project)

## Dataset

A detailed example of how to generate your data in parallel with PyTorch: <a href="https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel">link</a>

- preprocessing the data (take from covnet)  --> done in create_COVIDx_v3.ipynb (Covid Net)
test_COVIDx2.txt: This file contains the samples used for testing COVIDNet-CXR.)

- data loader + augmentation (PyTorch)

- data augmentation was leveraged with the following augmentation types: translation, rotation, horizontal flip, and intensity shift and batch re-balancing strategy to promote better distribution of each infection type at a batch level  --> done in data.py

## Network architecture
-COVID-NET has to following architecture:
![COVID-Net Architecture.](images/COVID-NET.png)

- I used the following formula to discover the padding and stride for each step:

<p align="center">
  <img width="180" height="60" src="images/equation_cnn_out.png">
</p>

Which leads us to stride = 2 and padding = 3 when applying the Conv7x7 and stride =2 and padding = 0 for the Conv1x1.

- Inside the PEPX, we have a DWConv3x3. This is a convolution that doesn’t change the format of the input:
<p align="center">
  <img width="420" height="300" src="images/Depthwise_convolution.png">
</p>

Links for better understand the architecture:

Depthwise-separable-convolutions: <a href="https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/"> link </a> 

When we have muldiple dimentions on image of feature map: <a href="https://www.researchgate.net/post/How_will_channels_RGB_effect_convolutional_neural_network"> link </a> 

1x1 convolutions : <a href="https://machinelearningmastery.com/introduction-to-1x1-convolutions-to-reduce-the-complexity-of-convolutional-neural-networks/"> link </a> 


## Training
- Pretrained on ImageNet dataset
- Trained on COVIDx dataset using Adam optimizer with a learning rate policy where the learning rate decreases when learning  stagnates for a period of time (i.e., ’patience’)
- The following hyperparameters were used for training: learning rate=2e-5, number of epochs=22, batch size=8, factor=0.7, patience=5

-->  done in train_tf.py

## Evaluating results
### Quantitative: 
- computed the test accuracy, as well as sensitivity and positive predictive value (PPV) for each infection type, on the aforementioned COVIDx dataset
- sensitivity and PPV for each infection type
- confusion matrix 

--> done in eval.py (Covid Net)


## Improvements / Changes to try out
- Try different loss structures --> importance sampler loss
- augmentation techniques
- use data generation techniques --> VAE
- learn the latent space --> VAE
- can we treat lack of data as missing data? 
- EM and Variational approaches are possible? 





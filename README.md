# nndetails

Getting the Neural Network layers, memory and parameters in PyTorch. The output is a table with these contents and it will print the table in a .txt file. Additionally, the filename can be specified.

## Example Usage with PyTorch

```
from nndetails import nndetails
import torchvision
cnn = torchvision.models.densenet201()
nndetails(cnn,(1,3,224,224))
```
For specific filenames
```
nndetails(cnn,(1,3,224,224),"densenet201")
```

## Additional Notes

The input size is defined to be (N,C,H,W) , and it is usually an image in conjunction with most Convolutional Neural Networks(CNN).
* N is the number of images
* C is the number of channels (usually 3 for an RGB image)
* H is the height of the image in pixels
* W is the width of the image in pixels

Input Sizes for the predefined models in torchvision.models:

CNN Model  | Example Input Size
------------- | -------------
AlexNet  | (1,3,224,224)
DenseNet  | (1,3,224,224)
Inception | (2,3,299,299)
ResNet | (1,3,224,224)
SqueezeNet | (1,3,224,224)
VGG | (1,3,224,224)

## Limitations

Most neural networks are defined for a specific input size or a range of input sizes, therefore the input size must be known before trying to execute this function. 

## Bugs

For predefined models in torchvision.models of resnet50, resnet101, and resnet152, there are 2 additional layers of Conv2d and BatchNorm2d between layer1.0.bn3 and layer1.0.relu. # TO FIX

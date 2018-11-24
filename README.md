# nndetails

Getting the Neural Network layers, memory and parameters in PyTorch. The output is a table with these contents and it will print the table in a .txt file in the current working directory. Additionally, the filename can be specified.

## Prerequistes
Install prettytable:
```
pip install PrettyTable
```

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

The layers can be taken from torch.nn and torch.nn.functional. The function, nndetails, can only take the layers from torch.nn not from torch.nn.functional. Therefore not all the layers will be detected. Additionally, it only takes the forward function because it uses a forward hook to get the layers. Layers which are defined in \_\_init\_\_ are sometimes not all present in the forward function, depending on how the network is defined.

Unfortunately, it uses the protected member \_modules and \_forward\_hooks

## Bugs

For predefined models in torchvision.models of resnet50, resnet101, and resnet152, the layer names do not match. The forward layers and the layers predefined in \_\_init\_\_ are different in torchvision.models.resnet

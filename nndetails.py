import os
import numbers
from prettytable import PrettyTable, FRAME
from numpy import prod, size
import torch
from torch.autograd import Variable


def nndetails(neural_network, input_size, *args):
    """Gets the details of the neural network defined by nn.Module
    Args:
        neural_network: is a class of neural_network.Module
        input_size:     must be a tuple (enclosed in brackets) of 4 dimensions,
                        e.g. (1,3,224,224) <-- use this for most
                        predefined models in torchvision.models
        *args:          is the filename in .txt which only accepts one value of
                        type string, e.g. "test" or 'test'.
                        The default is the class's name which may be
                        overwritten when different models use the same class

    Example Usage:
        import torchvision
        cnn = torchvision.models.resnet18()
        nndetails(cnn,(1,3,224,224))
    """
    # Nonlocal Variables
    input_sizes = []
    output_sizes = []
    memory = []
    weight_size = []
    weights = []
    bias = []
    layer_type_forward = []

    # Local Variables
    layer_name = []
    layer_type_init = []

    def printnorm(self, inp, output):
        """The forward hook function"""
        # Nonlocal Variables
        nonlocal input_sizes
        nonlocal output_sizes
        nonlocal memory
        nonlocal weight_size
        nonlocal weights
        nonlocal bias
        nonlocal layer_type_forward

        # Input Sizes, Output Sizes, and Memory
        input_sizes.append(inp[0].size())
        output_sizes.append(output.data.size())
        # (prod(output.data.size())*4)
        memory.append(output.data.numpy().nbytes)

        # Weights
        if hasattr(self, 'weight'):
            if self.weight is not None:
                weight_size.append(self.weight.size())
                weights.append(prod(self.weight.size()))
            else:
                weight_size.append(0)
        else:
            weight_size.append('')
            weights.append('')

        # Bias
        if hasattr(self, 'bias'):
            if self.bias is not None:
                bias.append(prod(self.bias.size()))
            else:
                bias.append(0)
        else:
            bias.append('')

        # Layer Type
        layer_type_forward.append(self.__class__.__name__)

    for name, i in list(neural_network.named_modules()):
        # Gets all individual layers in the network
        if not i._modules.keys():

            # Register forward hooks
            i.register_forward_hook(printnorm)

            # Get the layer names and type for later comparison
            # This is because, sometimes the network defined in __init__ does
            # not specify all the layers that are in the forward function
            # such as in torchvision.models.resnet
            layer_type_init.append(i.__class__.__name__)
            layer_name.append(name)

    # Run through the neural network with a specified input size
    # Error Handling (Getting rid of the forward hooks)
    try:
        neural_network(Variable(torch.randn(input_size)))

        # layer types comparison and assigning layer names to the correct layer
        for i in enumerate(layer_type_forward):
            if layer_type_forward[i[0]] != layer_type_init[i[0]]:
                layer_name.insert(i[0], '')
                layer_type_init.insert(i[0], '')

    except RuntimeError:
        # If there is an error do the following: Remove forward hooks
        for name, i in list(neural_network.named_modules()):
            if not i._modules.keys():
                i._forward_hooks.clear()

        # Most common fix
        print('Error: Input Size must be changed to fit the neural network')
        raise  # Get all the details of the error for inspection
        return

    # Length Check
    # print(len(layer_name),len(layer_type_forward),len(input_sizes),len(output_sizes),len(memory),len(weight_size),len(weights),len(bias))

    # Construct a table using prettytable
    table = PrettyTable(['No.',
                         'Layer Name',
                         'Layer Type',
                         'Input Sizes',
                         'Output Sizes',
                         'Memory (Bytes)',
                         'Weight Sizes',
                         'Weights',
                         'Bias'])
    table.hrules = FRAME
    table.align = "c"

    for pos in range(size(layer_type_forward)):
        table.add_row([pos,
                       layer_name[pos],
                       layer_type_forward[pos],
                       input_sizes[pos],
                       output_sizes[pos],
                       memory[pos],
                       weight_size[pos],
                       weights[pos],
                       bias[pos]])

    total_weights = sum(num for num in weights
                        if isinstance(num, numbers.Number))
    total_bias = sum(num for num in bias
                     if isinstance(num, numbers.Number))

    table.add_row(['',
                   '-----',
                   '-----',
                   '-----',
                   '-----',
                   '-----',
                   '-----',
                   '-----',
                   '-----'])

    table.add_row(['',
                   '',
                   '',
                   '',
                   'Total Memory (Forward & Backward)',
                   '%d*2' % (sum(memory)),
                   'Total (Weights, Bias)',
                   total_weights,
                   total_bias])

    table.add_row(['',
                   'Total Number of Layers',
                   len(layer_type_forward),
                   '',
                   'Total Memory/Image',
                   '%.2fMB' % (sum(memory) * 2 / 1000000),
                   'Total Parameters',
                   total_weights + total_bias,
                   ''])

    print(table)

    # Saving to text file
    table_txt = table.get_string()

    if args is not None:
        fpath = os.getcwd() + '\\' + args[0] + '.txt'
    else:
        fpath = os.getcwd() + '\\' + neural_network.__class__.__name__ + ' Details.txt'

    with open(fpath, 'w') as file:
        file.write(table_txt)

    # Remove all forward hooks
    for name, i in list(neural_network.named_modules()):
        if not i._modules.keys():
            i._forward_hooks.clear()

    return

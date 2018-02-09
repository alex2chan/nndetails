import os
import numbers
from prettytable import PrettyTable,FRAME
from numpy import prod,size
import torch
from torch.autograd import Variable    

"""Gets the details of the neural network defined by nn.Module
Args:
    NN:          is a class of nn.Module
    input_size:  must be a tuple (enclosed in brackets) of 4 dimensions:
                 e.g. (1,3,224,224) <-- use this for most predefined models in torchvision.models
    *args:       is the filename in .txt which only accepts one value of type string, e.g. "test" or 'test'.
                 The default is the class's name which may be overwritten when different models use the same class

Example Usage: 
    import torchvision
    cnn = torchvision.models.resnet18()
    nndetails(cnn,(1,3,224,224))
"""
# Main function
def nndetails(NN,input_size,*args):

    # Nonlocal Variables
    input_sizes = []
    output_sizes = []
    memory = []
    weight_size = []
    weights = []
    bias = []
    layer_type_forward = [] # layers in the forward function
    
    # Local Variables
    layer_name = []
    layer_type_init = [] # layers defined in __init__

    # The forward hook function
    def printnorm(self, input, output):
        
        # Nonlocal Variables
        nonlocal input_sizes
        nonlocal output_sizes
        nonlocal memory
        nonlocal weight_size
        nonlocal weights
        nonlocal bias
        nonlocal layer_type_forward
        
        # Input Sizes, Output Sizes, and Memory
        input_sizes.append(input[0].size())
        output_sizes.append(output.data.size())
        memory.append(output.data.numpy().nbytes) # same as (prod(output.data.size())*4)
        
        # Weights
        if hasattr(self,'weight'):
            if self.weight is not None:
                weight_size.append(self.weight.size())
                weights.append(prod(self.weight.size()))
            else:
                weight_size.append(0)
        else:
            weight_size.append('')
            weights.append('')
            
        # Bias
        if hasattr(self,'bias'):
            if self.bias is not None:
                bias.append(prod(self.bias.size()))
            else:
                bias.append(0)
        else:
            bias.append('')
            
        # Layer Type
        layer_type_forward.append(self.__class__.__name__)
        

    for count, (name, i) in enumerate(list(NN.named_modules())):
        if len(i._modules.keys()) == 0: # Gets all individual layers in the network

            # Register forward hooks to get input size, output size and memory for each layer
            i.register_forward_hook(printnorm)
            
            # Get the layer names and type for later comparison
            # This is because, sometimes the network defined in __init__ does not specify all the layers that are in the forward function
            # such as in torchvision.models.resnet
            layer_type_init.append(i.__class__.__name__)
            layer_name.append(name)
            

    # Run through the neural network with a specified input size
    # Error Handling (Getting rid of the forward hooks)
    try:
        NN(Variable(torch.randn(input_size)))
        
        # layer types comparison and assigning layer names to the correct layer
        for i in range(len(layer_type_forward)):
            if layer_type_forward[i] != layer_type_init[i]:
                layer_name.insert(i,'')
                layer_type_init.insert(i,'')
        
    except RuntimeError:
        # If there is an error do the following: Remove forward hooks
        for name, i in list(NN.named_modules()):
            if len(i._modules.keys()) == 0:
                i._forward_hooks.clear()
        
        # Most common fix
        print('Error: Input Size must be changed to fit the neural network')
        raise # Get all the details of the error for inspection
        return
    
    # Length Check
    #print(len(layer_name),len(layer_type),len(input_sizes),len(output_sizes),len(memory),len(weight_size),len(weights),len(bias))
    
    # Construct a table using prettytable
    t = PrettyTable(['Layer Name','Layer Type','Input Sizes','Output Sizes','Memory (Bytes)','Weight Sizes','Weights','Bias'])
    t.hrules = FRAME
    t.align = "c"

    for p in range(size(layer_type_forward)):
        t.add_row([layer_name[p],layer_type_forward[p],input_sizes[p],output_sizes[p],memory[p],weight_size[p],weights[p],bias[p]])
        
    total_weights = sum(num for num in weights if isinstance(num,numbers.Number))
    total_bias = sum(num for num in bias if isinstance(num,numbers.Number))
    
    t.add_row(['-----','-----','-----','-----','-----','-----','-----','-----'])
    t.add_row(['','','','Total Memory (Forward & Backward)','%d*2' %(sum(memory)),'Total (Weights, Bias)',total_weights,total_bias])
    t.add_row(['Total Number of Layers',len(layer_type_forward),'','Total Memory/Image','%.2fMB' %(sum(memory)*2/1000000),'Total Parameters',total_weights + total_bias,''])
    print(t)

    # Saving to text file
    table_txt = t.get_string()
    
    if len(args) > 0:
        fpath = os.getcwd() + '\\' + args[0] + '.txt'
    else:
        fpath = os.getcwd() + '\\' + NN.__class__.__name__ + ' Details.txt'
    
    with open(fpath,'w') as file:
        file.write(table_txt)

    # Remove all forward hooks
    for name, i in list(NN.named_modules()):
        if len(i._modules.keys()) == 0:
            i._forward_hooks.clear()

    return

"""This file defines the CrowdCount model which is a pytorch model.

The model is based on the Faster R-CNN architecture but has a simpler
output layer since we are only counting the number of people in an image.
"""
import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net

class CrowdCount(nn.Module):
    """The CrowdCount model.
    
    This model is based on the Faster R-CNN architecture but has a simpler
    output layer since we are only counting the number of people in an image.
    """
    def __init__(self, load_weights=False):
        """Initialize the CrowdCount model.
        
        Arguments:
            load_weights (bool, optional): If True, the model will be initialized
                with the weights of a pre-trained VGG16 model. Defaults to False.
        """
        super(CrowdCount, self).__init__()
        # The number of images the model has seen during training.
        self.seen = 0
        # The feature extractor of the model (i.e. the CNN part of the VGG16
        # architecture).
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # The feature pyramid network (i.e. the part of the network after the
        # feature extractor).
        self.backend_feat  = [512, 512, 512,256,128,64]
        # Create the feature extractor.
        self.frontend = make_layers(self.frontend_feat)
        # Create the feature pyramid network.
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        # The output layer of the model.
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        # If we are not loading pre-trained weights, initialize the weights of
        # the model to reasonable default values.
        if not load_weights:
            # Create a pre-trained VGG16 model.
            mod = models.vgg16(pretrained = True)
            # Initialize the weights of the model to reasonable default values.
            self._initialize_weights()
            # Loop over all the weights of the feature extractor and assign the
            # corresponding weights from the pre-trained model to the feature
            # extractor of the CrowdCount model.
            for i in range(len(self.frontend.state_dict().items())):
                # The name of the parameter in the CrowdCount model.
                k_crowd = list(self.frontend.state_dict().items())[i][0]
                # The value of the parameter in the CrowdCount model.
                v_crowd = list(self.frontend.state_dict().items())[i][1]
                # The name of the parameter in the pre-trained model.
                k_vgg = list(mod.state_dict().items())[i][0]
                # The value of the parameter in the pre-trained model.
                v_vgg = list(mod.state_dict().items())[i][1]
                # If the names of the parameters are the same, assign the value
                # of the pre-trained parameter to the corresponding parameter in
                # the CrowdCount model.
                if k_crowd == k_vgg:
                    v_crowd.data[:] = v_vgg.data[:]
    def forward(self,x):
        """Forward pass through the model.
        
        Arguments:
            x (torch.Tensor): The input image.
        
        Returns:
            The predicted number of people in the image.
        """
        x = self.frontend(x)
        """Pass the input image through the frontend (i.e. feature extractor)."""
        x = self.backend(x)
        """Pass the output of the frontend through the backend (i.e. feature pyramid network)."""
        x = self.output_layer(x)
        """Pass the output of the backend through the output layer (i.e. count prediction layer)."""
        return x
    def _initialize_weights(self):
        """Initialize the weights of the model."""
        for m in self.modules():
            """Iterate over all the layers in the model."""
            if isinstance(m, nn.Conv2d):
                """If the layer is a convolutional layer."""
                nn.init.normal_(m.weight, std=0.01)
                """Initialize its weights to have a standard deviation of 0.01."""
                if m.bias is not None:
                    """If the layer has a bias (i.e. is not a convolutional layer without a bias)."""
                    nn.init.constant_(m.bias, 0)
                    """Initialize its bias to 0."""
            elif isinstance(m, nn.BatchNorm2d):
                """If the layer is a batch normalization layer."""
                nn.init.constant_(m.weight, 1)
                """Initialize its gamma (i.e. weight) to 1."""
                nn.init.constant_(m.bias, 0)
                """Initialize its beta (i.e. bias) to 0."""
                    
def make_layers(cfg, in_channels = 3, batch_norm=False, dilation = False):
    """Builds a layer based on the specified configuration.

    Arguments:
        cfg (list): A list of integers, where each integer represents the number of output
            channels in the corresponding convolutional layer. 'M' stands for a max pooling
            layer, which halves the spatial dimensions of the input.
        in_channels (int, optional): The number of input channels. Defaults to 3.
        batch_norm (bool, optional): If True, include batch normalization after each
            convolutional layer. Defaults to False.
        dilation (bool, optional): If True, use a dilation rate of 2 in each convolutional
            layer. Defaults to False.
        inplace (bool, optional): If True, use in-place operations whenever possible. This
            can slightly reduce memory usage, but may negatively impact performance. Defaults
            to False.

    Returns:
        nn.Sequential: A sequential container of layers specified in the configuration.
    """
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            """
            If the current value is 'M', add a max pooling layer with a kernel size of 2 and a
            stride of 2 to the list of layers. This effectively halves the spatial dimensions of
            the input.
            """
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            """
            Otherwise, add a convolutional layer with the specified number of output channels.
            The kernel size is fixed to 3, and the padding is set to the dilation rate to avoid
            losing pixels when downsampling.
            """
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation = d_rate)
            if batch_norm:
                """
                If batch normalization is enabled, add it after the convolutional layer.
                """
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                """
                Otherwise, just add a ReLU activation function after the convolutional layer.
                """
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers) 


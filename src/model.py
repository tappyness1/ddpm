import numpy as np
import torch
import torch.nn as nn
from torch.nn import (AdaptiveAvgPool2d, BatchNorm2d, Conv2d, ConvTranspose2d,
                      Linear, MaxPool2d, ReLU, Softmax)
from src.conv_size_calc import get_output_size


def reparameterise(mean, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return mean + std*eps

class VAE(nn.Module):
    def __init__(self, cfg_obj):
        super().__init__()
        self.cfg_obj = cfg_obj

        # get the encoder and decoder
        # output of the encoder will then go through another 2 separate FC
        # first is to get mu, the mean
        # second is to get the variance
        # for numerical stability, we will obtain the log variance instead
        # read more here - https://stats.stackexchange.com/questions/353220/why-in-variational-auto-encoder-gaussian-variational-family-we-model-log-sig

        if cfg_obj["dataset"]['dataset'] in ["FashionMNIST", "MNIST"]:
            self.encoder_decoder = EncoderDecoderMNIST()

        if cfg_obj["dataset"]['dataset'] == "Flowers102":
            # in the case of Flowers102, which are 3x224x224 images, we do conv layers then flatten
            self.encoder_decoder = EncoderDecoderFlowers102()

        # some papers say that your sigma can be a trainable parameter
        # some say you can just set it to 1 or any number you want. 
        self.log_scale = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, input):

        return self.encoder_decoder(input)
    
class EncoderDecoderMNIST(nn.Module):
    def __init__(self, encoder_params=[28*28, 512, 256], latent_dim = 128):
        super(EncoderDecoderMNIST, self).__init__()
        relu = nn.ReLU()
        # encoder_params = [28*28, 512, 256, 128, 64, 32, 16]
        self.encoder = nn.Sequential()
        for i in range(len(encoder_params)-1):
            self.encoder.add_module(f"encoder_{i}", nn.Linear(encoder_params[i], encoder_params[i+1]))
            self.encoder.add_module(f"encoder_relu_{i}", relu)
        self.decoder = nn.Sequential()
        self.decoder.add_module(f"decoder", nn.Linear(latent_dim, encoder_params[-1]))
        for i in range(len(encoder_params)-1, 0, -1):
            self.decoder.add_module(f"decoder_{i}", nn.Linear(encoder_params[i], encoder_params[i-1]))
            self.decoder.add_module(f"decoder_relu_{i}", relu)

        self.FC_mean = nn.Linear(encoder_params[-1], latent_dim)
        self.FC_logvar = nn.Linear(encoder_params[-1], latent_dim)

    def forward(self, input):
        # input = input.reshape(-1, 28*28)
        input = input.view(-1, 28*28)
        encoded = self.encoder(input)
        mean = self.FC_mean(encoded)
        log_var = self.FC_logvar(encoded)

        z = reparameterise(mean, log_var)

        decoded = self.decoder(z)
        decoded = decoded.reshape(-1, 1, 28, 28)
        decoded = decoded.view(-1,1,28,28)
        return mean, log_var, z, decoded
    
class EncoderDecoderFlowers102(nn.Module):
    def __init__(self, img_size = 224) -> None:
        super(EncoderDecoderFlowers102, self).__init__()
        relu = nn.ReLU()
        stride = 2
        out_paddings = [1,0,0]
        self.encoder = nn.Sequential()
        encoder_conv_params = [3,32,64]
        self.encoder_conv_params = encoder_conv_params 
        self.encoder_out_size = get_output_size(img_size, 3, stride = stride, num_times= len(encoder_conv_params)-1)
        encoder_lin_params = [self.encoder_out_size*self.encoder_out_size*encoder_conv_params[-1], 1024, 512]

        for i in range(len(encoder_conv_params)-1):
            self.encoder.add_module(f"encoder_{i}", nn.Conv2d(encoder_conv_params[i], encoder_conv_params[i+1], 3, stride = 2))
            self.encoder.add_module(f"encoder_relu_{i}", relu)

        self.encoder.add_module("flatten", nn.Flatten())

        for i in range(len(encoder_lin_params)-1):
            self.encoder.add_module(f"encoder_lin_{i}", nn.Linear(encoder_lin_params[i], encoder_lin_params[i+1]))
            self.encoder.add_module(f"encoder_lin_relu_{i}", relu)

        self.FC_mean = nn.Linear(512 , 512)
        self.FC_logvar = nn.Linear(512 , 512)

        self.decoder_linear = nn.Sequential()
        for i in range(len(encoder_lin_params)-1, 0, -1):
            self.decoder_linear.add_module(f"decoder_lin_{i}", nn.Linear(encoder_lin_params[i], encoder_lin_params[i-1]))
            self.decoder_linear.add_module(f"decoder_lin_relu_{i}", relu)
        
        self.decoder_conv = nn.Sequential()
        for i in range(len(encoder_conv_params)-1, 0, -1):
            self.decoder_conv.add_module(f"decoder_{i}", nn.ConvTranspose2d(encoder_conv_params[i], encoder_conv_params[i-1], 3, stride = stride, output_padding= out_paddings[i-1]))
            self.decoder_conv.add_module(f"decoder_relu_{i}", relu)

    def forward(self, input):        
            
        encoded = self.encoder(input)
        mean = self.FC_mean(encoded)
        log_var = self.FC_logvar(encoded)
        z = reparameterise(mean, log_var)

        decoded = self.decoder_linear(z)
        decoded = decoded.reshape(-1, self.encoder_conv_params[-1], self.encoder_out_size, self.encoder_out_size)
        decoded = self.decoder_conv(decoded)
        
        return mean, log_var, z, decoded

if __name__ == "__main__":
    import numpy as np
    import torch
    from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 3, 224, 224).astype('float32')
    
    X = torch.tensor(X)

    cfg_obj = {"dataset": {"dataset":"Flowers102"}}
    model = VAE(cfg_obj = cfg_obj)
    
    # summary(model, (3, 224, 224))
    print ()
    print (model.forward(X)[0].shape)
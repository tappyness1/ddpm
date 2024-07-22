import torch
import numpy as np
from src.model import VAE

def recon_loss(decoded, x):
    mse = torch.nn.MSELoss(reduction='sum')
    return mse(decoded, x)

    # bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
    # return bce(decoded, x)

def kl_divergence(mean, log_var):
    # do it like they do in the paper.
    kl = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())
    return kl

def elbo_loss(mean, log_var, decoded, x):
    r_loss = recon_loss(decoded, x)

    kl_loss = kl_divergence(mean, log_var)

    return kl_loss + r_loss

if __name__ == "__main__":
    from src.model import VAE
    from src.dataset import get_load_data

    cfg_obj = {"dataset": {"dataset":"FashionMNIST"}}
    dataset = cfg_obj['dataset']['dataset']
    vae_model = VAE(cfg_obj)
    
    train, test = get_load_data(root = "../data", dataset = dataset, download = False)
      
    img, label = train[0] 
    
    img = img.unsqueeze(0)
    mean, log_var, z, decoded = vae_model(img)
    loss = elbo_loss(mean, log_var, z, vae_model.log_scale, decoded, dataset, img)
    loss.backward()
    print (loss)
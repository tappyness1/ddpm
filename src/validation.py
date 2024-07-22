import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
from src.loss_function import elbo_loss
from sklearn.metrics import confusion_matrix

def validation(model, val_set, cfg_obj):
    """Simple validation workflow. Current implementation is for F1 score

    Args:
        model (_type_): _description_
        val_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    val_dataloader = DataLoader(val_set, batch_size=5, shuffle = True)

    dataset = cfg_obj['dataset']['dataset']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    losses = []

    with tqdm(val_dataloader) as tepoch:

        for imgs, _ in tepoch:
            imgs = imgs.to(device)
            with torch.no_grad():
                mean, log_var, z, decoded = model(imgs)

            loss = elbo_loss(mean, log_var, z, model.log_scale, decoded, dataset, imgs)  
            tepoch.set_postfix(loss=loss.item())  
            losses.append(loss.item())

    print (f"Validation Loss: {sum(losses)/len(losses)}")

    return sum(losses)/len(losses)


if __name__ == "__main__":
    
    from src.dataset import get_load_data


    cfg = {"save_model_path": "model_weights/model_weights.pt",
           'show_model_summary': True, 
           'train': {"epochs": 1, 'lr': 0.005, 'weight_decay': 5e-3},
           'dataset': {"dataset": "Flowers102"}}
    
    _, val_set = get_load_data(root = "../data", dataset = cfg['dataset']['dataset'])

    trained_model_path = "model_weights/model_weights.pt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(trained_model_path, map_location=torch.device(device))
    validation(model, val_set, cfg_obj=cfg)
            
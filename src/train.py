import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from src.dataset import get_load_data
from src.loss_function import elbo_loss
from src.model import VAE

def train(train_set, cfg):
           
    model = VAE(cfg_obj = cfg)

    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    if cfg['show_model_summary']:
        summary(model, (3,224,224))

    train_dataloader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle = True)
    
    # dataset = cfg['dataset']['dataset']
    dataset_name = cfg['dataset']['dataset'].lower()

    num_epochs = cfg['train']['epochs']

    if cfg['train']['continue_training']:
        model_weights = torch.load(cfg['train']['weights_path'],
                                   map_location=torch.device(device))
        model.load_state_dict(model_weights)

        # get the saved epochs and continue training from there
        last_epoch = int(cfg['train']['weights_path'].split("-")[-1].split(".")[0])
        num_epochs -= last_epoch

    last_epoch = last_epoch if cfg['train']['continue_training'] else 0
    
    for epoch in range(last_epoch+1, last_epoch + num_epochs+1):
        print (f"Epoch {epoch}:")
        # for i in tqdm(range(X.shape[0])):
        overall_loss = []
        with tqdm(train_dataloader) as tepoch:
            for imgs, classes in tepoch:
                imgs = imgs.to(device)
                
                mean, log_var, z, decoded = model(imgs)
                loss = elbo_loss(mean, log_var, decoded, imgs)
                
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())
                tepoch.set_postfix(loss=loss.item())
                
            print(f"Epoch {epoch} loss: {sum(overall_loss) / (len(overall_loss)*cfg['train']['batch_size'])}")

        if epoch % cfg['train']['save_checkpoint_interval'] == 0:
            torch.save(model.state_dict(), f"{cfg['save_model_path']}-{dataset_name}-epochs-{epoch}.pt")
            print (f"Checkpoint saved at epoch {epoch}")

    print("training done")
    torch.save(model.state_dict(), f"{cfg['save_model_path']}-{dataset_name}.pt")

    return model

if __name__ == "__main__":

    torch.manual_seed(42)

    # cfg = {"save_model_path": "model_weights/vae",
    #        'show_model_summary': False, 
    #        'train': {"epochs": 50, 'lr': 0.001, 'weight_decay': 5e-3, 
    #                 'batch_size': 4096},
    #        'dataset': {"dataset": "FashionMNIST"}} # FashionMNIST, MNIST

    cfg = {"save_model_path": "model_weights/vae",
           'show_model_summary': False, 
           'train': {"epochs": 100, 'lr': 0.001, 'weight_decay': 5e-3, 'batch_size': 8, 
                     'continue_training': False, 
                     'save_checkpoint_interval': 10,
                     'weights_path': "model_weights/vae-flowers102-epochs-20.pt"},
           'dataset': {"dataset": "Flowers102"}}  

    # train_set, _ = get_load_data(root = "../data", dataset = cfg['dataset']['dataset'])
    train_set, _ = get_load_data(root = "/content/data", dataset = cfg['dataset']['dataset'], download = True) # for colab
    train(train_set = train_set, cfg = cfg)    
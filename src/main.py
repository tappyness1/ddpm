from src.train import train
from src.validation import validation
import hydra
from omegaconf import DictConfig, OmegaConf
from src.dataset import get_load_data
import torch

@hydra.main(version_base = None, config_path="../cfg", config_name="cfg")
def train_model(cfg : DictConfig):

    torch.manual_seed(42)
    train_set, test_set = get_load_data(root = cfg["dataset"]["root"], dataset = cfg["dataset"]["dataset"])
    network = train(train_set = train_set, cfg = cfg)
    loss = validation(network, test_set, cfg_obj=cfg)

    return loss

@hydra.main(version_base = None, config_path="../cfg", config_name="cfg_hptuning.yaml")
def train_model_hptuning(cfg : DictConfig):

    torch.manual_seed(42)
    train_set, test_set = get_load_data(root = cfg["dataset"]["root"], dataset = cfg["dataset"]["dataset"])
    network = train(train_set = train_set, cfg = cfg)
    loss = validation(network, test_set)

    return loss

if __name__ == "__main__":
    train_model()
    # train_model_hptuning() # remember to use --multirun flag if it is not turned on in the yaml

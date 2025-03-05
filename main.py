import torch
import yaml
import gc
import warnings
import random
from utils import *
from tqdm import tqdm
import wandb
from datetime import datetime
import argparse
from torch.utils.data import DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler

from trainer import Trainer 
# from MelGAN import Discriminator_MelGAN
# from MBSTFTD import MultiBandSTFTDiscriminator

## models
from models.model import MBSEANet
from models.model_tfilm import MBSEANet_film
from models.model_tfilm_sbr import MBSEANet_film_sbr
from models.model_tfilm_core import MBSEANet_film as MBSEANet_film_core
from models.model_pqmf import MBSEANet_pqmf 

from models.prepare_models import prepare_generator, prepare_discriminator
## dataset
from dataset import CustomDataset
# from trainer import Trainer

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE: {DEVICE}")

TIME = datetime.now()
print(TIME.strftime("%Y-%m-%d %H:%M:%S"))

# Dictionary for models and configuration
MODEL_MAP = {
    "MBSEANet": MBSEANet,
    "MBSEANet_film": MBSEANet_film,
    "MBSEANet_film_sbr": MBSEANet_film_sbr,
    "MBSEANet_film_core": MBSEANet_film_core,
    "MBSEANet_pqmf": MBSEANet_pqmf,
}

def parse_args():
    parser = argparse.ArgumentParser(description="mbseanet Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--wandb', type=lambda x:x.lower() == 'true', required=True, help="wandb logging (True/False)")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def prepare_dataloader(config_path):
    """
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    config = load_config(config_path)
    
    # Load datasets
    train_dataset = CustomDataset(
        path_dir_nb=config['dataset']['nb_train'], 
        path_dir_wb=config['dataset']['wb_train'], 
        seg_len=config['dataset']['seg_len'], 
        mode="train",
        start_index=config['dataset']['start_index'], 
        high_index=config['dataset']['high_index'],
        use_sfm=config['dataset']['use_sfm'],
        use_pqmf=config['dataset'].get('use_pqmf_features', 0)
    )
    

    val_dataset = CustomDataset(
        path_dir_nb=config['dataset']['nb_test'], 
        path_dir_wb=config['dataset']['wb_test'], 
        seg_len=config['dataset']['seg_len'], 
        mode="val",
        start_index=config['dataset']['start_index'], 
        high_index=config['dataset']['high_index'],
        use_sfm=config['dataset']['use_sfm'],
        use_pqmf=config['dataset'].get('use_pqmf_features', 0)
    )
    
    # Optionally split train data
    if config['dataset']['ratio'] < 1:
        train_size = int(config['dataset']['ratio'] * len(train_dataset))
        _, train_dataset = random_split(train_dataset, [len(train_dataset) - train_size, train_size])

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=True,
        num_workers=config['dataset']['num_workers'], 
        prefetch_factor=2,
        persistent_workers=True, 
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=config['dataset']['num_workers'], 
        prefetch_factor=2, 
        persistent_workers=True, 
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader


def main(if_log_step):
    args = parse_args()
    if_log_to_wandb = args.wandb
    config = load_config(args.config)
    torch.manual_seed(42)
    random.seed(42)
    
    if if_log_to_wandb: # if log
        wandb.init(project='mbseanet_a', entity='woongzip1', config=config, name=config['run_name'], notes=config['run_name'])
    
    print_config(config)
    
    # Prepare dataloader
    train_loader, val_loader = prepare_dataloader(args.config)
    
    # Model selection
    generator = prepare_generator(config, MODEL_MAP)
    discriminator = prepare_discriminator(config)

    # Optimizers
    if config['generator']['fe_weight_path']:
        print("------------Fine Tuning!------------")
        non_fe_params = [p for p in generator.parameters() if p not in set(generator.feature_encoder.parameters())]

        optim_G = torch.optim.Adam(
            [
                {'params': generator.feature_encoder.parameters(), 'lr': config['optim']['learning_rate_ft']}, 
                {'params': non_fe_params, 'lr': config['optim']['learning_rate']}  
            ],
            betas=(config['optim']['B1'], config['optim']['B2'])
        )
        optim_D = torch.optim.Adam(discriminator.parameters(), lr=config['optim']['learning_rate'], betas=(config['optim']['B1'], config['optim']['B2']))
    else: # scratch
        optim_G = torch.optim.Adam(generator.parameters(), lr=config['optim']['learning_rate'], betas=(config['optim']['B1'], config['optim']['B2']))
        optim_D = torch.optim.Adam(discriminator.parameters(), lr=config['optim']['learning_rate'], betas=(config['optim']['B1'], config['optim']['B2']))
        
    # Schedulers
    if config['use_tri_stage']:
        from scheduler import TriStageLRScheduler
        print("*** TriStageLRScheduler ***")
        scheduler_G = TriStageLRScheduler(optimizer=optim_G, **config['tri_scheduler'])
        scheduler_D = TriStageLRScheduler(optimizer=optim_D, **config['tri_scheduler'])
    else:
        print("*** Exp LRScheduler ***")
        scheduler_G = lr_scheduler.ExponentialLR(optim_G, gamma=config['optim']['scheduler_gamma'])
        scheduler_D = lr_scheduler.ExponentialLR(optim_D, gamma=config['optim']['scheduler_gamma'])

    import pdb
    pdb.set_trace()

    # Trainer initialization
    trainer = Trainer(generator, discriminator, train_loader, val_loader, optim_G, optim_D, config, DEVICE, 
                      scheduler_G=scheduler_G, scheduler_D=scheduler_D, if_log_step=if_log_step, if_log_to_wandb=if_log_to_wandb)
    
    if config['train']['ckpt']:
        trainer.load_checkpoints(config['train']['ckpt_path'])    
    
    torch.manual_seed(42)
    random.seed(42)
    
    # Train
    warnings.filterwarnings("ignore", category=UserWarning, message="At least one mel filterbank has")
    trainer.train(num_epochs=config['train']['max_epochs'])

if __name__ == "__main__":
    main(if_log_step=True)

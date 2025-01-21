""" forward files and save into inference directory """
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
import soundfile as sf

from trainer import Trainer 
# from MelGAN import Discriminator_MelGAN
# from MBSTFTD import MultiBandSTFTDiscriminator

## models
from models.model import MBSEANet
from models.model_tfilm import MBSEANet_film
from models.model_tfilm_sbr import MBSEANet_film_sbr
from models.prepare_models import prepare_generator, prepare_discriminator
## dataset
from dataset import CustomDataset
## main
from main import load_config, prepare_dataloader
## 
from pqmf import PQMF


# Dictionary for models and configuration
MODEL_MAP = {
    "MBSEANet": MBSEANet,
    "MBSEANet_film": MBSEANet_film,
    "MBSEANet_film_sbr": MBSEANet_film_sbr,
}

def load_model_params(model, checkpoint_path, device='cuda'):
    model = model.to(device)
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['generator_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model

def main(config, device, gt_base=None, save_files=True):
    save_base_dir = config['eval']['eval_dir_audio']
    os.makedirs(save_base_dir, exist_ok=True)

    _, val_loader = prepare_dataloader(args.config)
    model = prepare_generator(config, MODEL_MAP)
    model = load_model_params(model, config['train']['ckpt_path'], device=device)

    ## PQMF
    pqmf_ = PQMF(num_subbands=32, num_taps=481, cutoff_ratio=None).to(device) 

    torch.manual_seed(42)
    np.random.seed(42)

    ## forward
    model.eval()
    bar = tqdm(val_loader)

    with torch.no_grad():
        for hr, lr, cond, name, _ in bar:
            name = name[0]
            lr, hr, cond = lr.to(device), hr.to(device), cond.to(device)
            
            # import pdb
            # pdb.set_trace()

            ## Analysis
            nb = pqmf_.analysis(lr)[:, :config['generator']['c_in'], :] # core [B,5,T]
            hf_estimate, commitment_loss, codebook_loss = model(nb, cond)
            target_subbands = pqmf_.analysis(hr)[:, config['generator']['c_in']:, :] # target subbands [B,27,T] 

            ## BWE target
            # hr = pqmf_.synthesis(torch.cat((nb, target_subbands), dim=1), delay=0, length=hr.shape[-1]) # [B,T]

            x_hat = torch.cat((nb.detach(), hf_estimate), dim=1)
            x_hat_full = pqmf_.synthesis(x_hat, delay=0, length=hr.shape[-1])  # PQMF Synthesis

            if save_files:
                sf.write(f"{save_base_dir}/{name}.wav", x_hat_full.cpu().squeeze(), format="WAV", samplerate=48000)
                if gt_base is not None:
                    os.makedirs(gt_base, exist_ok=True)
                    sf.write(f"{gt_base}/{name}.wav", hr.cpu().squeeze(), format="WAV", samplerate=48000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with specified config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--device", type=str, required=True, default='cuda')
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device
    gt_base = "/home/woongjib/Projects/mbseanet_results/ground_truth/exp1_gt"
    
    main(config, device, gt_base=None, save_files=True)
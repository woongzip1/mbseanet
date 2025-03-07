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

from Projects.MBSEANet.trainer import Trainer 
# from MelGAN import Discriminator_MelGAN
# from MBSTFTD import MultiBandSTFTDiscriminator

## models
from models.model import MBSEANet
from models.model_tfilm import MBSEANet_film
from models.model_tfilm_sbr import MBSEANet_film_sbr
from models.model_tfilm_core import MBSEANet_film as MBSEANet_film_core
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
    "MBSEANet_film_core": MBSEANet_film_core,
}

from scipy.signal import firwin, lfilter, freqz
def lpf(y, sr=16000, cutoff=500, numtaps=200, window='hamming', figsize=(10,2)):
    """ 
    Applies FIR filter
    cutoff freq: cutoff freq in Hz

    """
    nyquist = 0.5 * sr
    normalized_cutoff = cutoff / nyquist
    taps = firwin(numtaps=numtaps, cutoff=normalized_cutoff, window=window)
    y_lpf = lfilter(taps, 1.0, y)
    # y_lpf = np.convolve(y, taps, mode='same')
    
    # Length adjust
    y_lpf = np.roll(y_lpf, shift=-numtaps//2)
    
    return y_lpf

def load_model_params(model, checkpoint_path, device='cuda'):
    model = model.to(device)
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['generator_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model

def main(config, device, gt_base=None, save_files=True, low_pass=False, cutoff=None, quant_n=None):
    save_base_dir = config['eval']['eval_dir_audio']
    if quant_n is not None:
        save_base_dir = f"{save_base_dir}_Q{quant_n}"
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
        for hr, lr, cond, name, sfm in bar:
            name = name[0]
            lr, hr, cond, sfm = lr.to(device), hr.to(device), cond.to(device), sfm.to(device)
            
            # import pdb
            # pdb.set_trace()

            ## Analysis
            nb = pqmf_.analysis(lr)[:, :config['generator']['c_in'], :] # core [B,5,T]
            if config['dataset']['use_sfm']:
                # print('SFM!')
                hf_estimate, commitment_loss, codebook_loss = model(nb, cond, sfm, n_quantizers=quant_n)                
            elif config['loss']['lambda_commitment_loss'] == 0:
                hf_estimate = model(nb, HR=None) 
            else:
                hf_estimate, commitment_loss, codebook_loss = model(nb, cond, None, n_quantizers=quant_n)
            target_subbands = pqmf_.analysis(hr)[:, config['generator']['c_in']:config['generator']['c_in']+config['generator']['c_out'], :] # target subbands [B,27,T] 

            ## BWE target
            _b,_f1,_l = nb.shape
            _b,_f2,_l = target_subbands.shape
            freq_zeros = torch.zeros(_b,32-_f1-_f2,_l).to(nb.device)
            # hr = pqmf_.synthesis(torch.cat((nb, target_subbands, freq_zeros), dim=1), delay=0, length=hr.shape[-1]) # [B,T]
            ###########
            
            x_hat = torch.cat((nb.detach(), hf_estimate, freq_zeros), dim=1)
            x_hat_full = pqmf_.synthesis(x_hat, delay=0, length=hr.shape[-1])  # PQMF Synthesis

            ## Low Pass Filter
            if low_pass:
                x_hat_full_lpf = lpf(x_hat_full.cpu().squeeze(), sr=48000, cutoff=cutoff,)
                os.makedirs(f"{save_base_dir}_lpf", exist_ok=True)

            if save_files:
                sf.write(f"{save_base_dir}/{name}.wav", x_hat_full.cpu().squeeze(), format="WAV", samplerate=48000)
            
                if low_pass:
                    sf.write(f"{save_base_dir}_lpf/{name}.wav", x_hat_full_lpf.squeeze(), format="WAV", samplerate=48000)
                    pass
                
                if gt_base is not None:
                    os.makedirs(gt_base, exist_ok=True)
                    # gt lpf save
                    hr = lpf(hr.cpu().squeeze(), sr=48000, cutoff=cutoff)
                    sf.write(f"{gt_base}/{name}.wav", hr.squeeze(), format="WAV", samplerate=48000)
                    # lr = lr.cpu().squeeze()
                    # sf.write(f"{gt_base}/{name}.wav", lr.squeeze(), format="WAV", samplerate=48000)
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with specified config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--device", type=str, required=True, default='cuda')
    parser.add_argument("--cutoff", type=int, required=True, )
    parser.add_argument("--quant_n", type=int, required=False, default=None)
    
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device
    gt_base = "/home/woongjib/Projects/mbseanet_results/ground_truth/core16"
    # gt_base None for not saving
    
    print(args.quant_n)
    main(config, device, gt_base=None, save_files=True, low_pass=False, cutoff=args.cutoff, quant_n=args.quant_n)
    
    """ python inference.py --config configs/exp4.yaml --device 'cuda' --cutoff 15500 """
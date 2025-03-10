import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
import os
import gc
from utils import draw_spec, lsd_batch

from matplotlib import pyplot as plt
import numpy as np
from loss import LossCalculator
from pqmf import PQMF
from torch.nn.utils import clip_grad_norm_

## changed into old ver (using GT as target)

class Trainer:
    def __init__(self, generator, discriminator, train_loader, val_loader, optim_G, optim_D, config, device, 
                 scheduler_G=None, scheduler_D=None, if_log_step=False, if_log_to_wandb=True):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.config = config
        self.device = device
        self.if_log_step = if_log_step
        self.if_log_to_wandb = if_log_to_wandb
        
        self.loss_calculator = LossCalculator(config, self.discriminator)
        self.lambda_commitment_loss = config['loss']['lambda_commitment_loss']
        self.lambda_codebook_loss = config['loss']['lambda_codebook_loss']
        self.lambda_mel_loss = config['loss']['lambda_mel_loss']
        self.lambda_fm_loss = config['loss']['lambda_fm_loss']
        self.lambda_adv_loss = config['loss']['lambda_adv_loss']
        

        self.use_sfm = config['dataset']['use_sfm']

        self.scheduler_G = scheduler_G
        self.scheduler_D = scheduler_D

        self.pqmf_fb = PQMF(num_subbands=32, num_taps=481, cutoff_ratio=None).to(device) 

        self.hr_logged = False
        self.start_epoch = 1
        self.start_step = 1
    
    def unified_log(self, log_dict, stage, epoch=None, step=None):
        """
        Unified logging function for wandb that handles different data types.

        Args:
            log_dict (dict): Dictionary containing log keys and values.
            stage (str): Training/validation stage ('train'/'val').
            epoch (int, optional): Epoch number for logging. Defaults to None.
            step: Step number for iterations. Used for logging in training
        """
        if self.if_log_to_wandb:
            for key, value in log_dict.items():
                if isinstance(value, torch.Tensor):
                    item = value.cpu().detach().numpy()
                else:
                    item = value
                
                try:
                    if isinstance(item, float) or isinstance(item, int):
                        log = item
                    elif isinstance(item, plt.Figure):  # Handling Matplotlib figures if applicable
                        log = wandb.Image(item)
                        plt.close(item)
                    elif isinstance(item, np.ndarray) and item.ndim in [2, 3]:  # Assuming this is an image
                        log = wandb.Image(item, caption=f"{stage.capitalize()} {key.capitalize()} Epoch {epoch}")
                    elif isinstance(item, np.ndarray) and item.ndim == 1:  # Assuming this is an audio signal
                        log = wandb.Audio(item, sample_rate=48000, caption=f"{stage.capitalize()} {key.capitalize()} Epoch {epoch}")
                    else:
                        log = item  # Default logging for non-special cases
                except Exception as e:
                    print(f"Failed to log {key}: {e}")
                    log = item  # Log as-is if an exception occurs

                wandb.log({
                    f"{stage}/{key}": log,
                }, step=epoch if epoch is not None else step)

    def _forward_pass(self, lr, cond, hr=None, sfm=None):
        if self.config['generator']['type'] == 'MBSEANet_pqmf':
            return self.generator(x=lr, hr=hr)
        
        if self.lambda_codebook_loss != 0:
            return self.generator(lr, cond, sfm) # three outputs
        return self.generator(lr, cond), 0, 0

    def train_step(self, hr, lr, cond, step, sfm=None, pretrain_step=0):
        self.generator.train()
        self.discriminator.train()
               
        ## analysis ##
        input_subbands, target_subbands = obtain_subbands(self.config, self.pqmf_fb, lr, hr)
    
        if self.use_sfm:
            hf_estimate, commitment_loss, codebook_loss = self._forward_pass(input_subbands, cond, sfm, hr=hr)
        else:
            hf_estimate, commitment_loss, codebook_loss = self._forward_pass(input_subbands, cond, hr=hr)        
        
        ## synthesis ##
        hr_target, x_hat_full = synthesize_subbands(hf_estimate, target_subbands, input_subbands, self.pqmf_fb, hr.shape[-1])
        
        if self.config['loss']['lambda_subband_loss'] == 0:
            hf_for_save = hf_estimate
            hf_estimate = None 
            
        loss_G, ms_mel_loss_value, g_loss_dict, g_loss_report, subband_loss_value, fullband_loss_value = self.loss_calculator.compute_generator_loss(
                                                                                        hr_target, x_hat_full, commitment_loss, codebook_loss,
                                                                                        hf_estimate=hf_estimate, target_subbands=target_subbands)  
        
        #### for gradient exploding ####
        if ms_mel_loss_value > 50:
            print("loss_G:", loss_G)
            print("loss_mel:", ms_mel_loss_value)
            print("commitment", commitment_loss)
            print("loss_G_dict", g_loss_dict)
            debug_data = {
                "lr": lr.detach().cpu(), "hr_target": hr_target.detach().cpu(),
                "input_subbands": input_subbands.detach().cpu(), "hf_estimate": hf_for_save.detach().cpu(),
                "input": x_hat_full.detach().cpu(), "target": hr_target.detach().cpu()}
            torch.save(debug_data, "debug/gradients.pt")
            print(f"Debug data saved to debug/")
            raise ValueError("Gradient Exploded!")        
        ########################
        
        # Train generator
        self.optim_G.zero_grad()
        loss_G.backward() 
        
        # gradient clip
        # if step < 2000:
        clip_grad_norm_(self.generator.parameters(), max_norm=3.0)
        
        self.optim_G.step()

        if step >= pretrain_step: # Train discriminator
            loss_D, d_loss_dict, d_loss_report = self.loss_calculator.compute_discriminator_loss(hr, x_hat_full)
            self.optim_D.zero_grad()
            loss_D.backward()
            self.optim_D.step()
        else: # only train generator (p)
            # print(f"PRETRAIN with {pretrain_step}")
            loss_D = 0
            d_loss_dict = {}
            d_loss_report = {}
            
        step_result = {
            'loss_G': loss_G.item(),
            'ms_mel_loss': ms_mel_loss_value.item() if ms_mel_loss_value else 0,
            # 'loss_D': loss_D.item(),
            **{f'G_{k}': v.item() if isinstance(v, torch.Tensor) else v for k, v in g_loss_dict.items()},
            **{f'D_{k}': v.item() if isinstance(v, torch.Tensor) else v for k, v in d_loss_dict.items()},
            **{f'G_report_{k}': v for k, v in g_loss_report.items()},  
            **{f'D_report_{k}': v for k, v in d_loss_report.items()},  
            'commitment_loss': commitment_loss.item() if commitment_loss else 0,
            'codebook_loss': codebook_loss.item() if codebook_loss else 0,
            # subband loss
            'subband_loss': subband_loss_value.item() if subband_loss_value else 0,
            'fullband_loss': fullband_loss_value.item() if fullband_loss_value else 0,
            
            }
        # import pdb
        # pdb.set_trace()
        if self.if_log_step and step % 100 == 0:
            self.unified_log(step_result, 'train', step=step)
            
        return step_result

    def validate(self, step=None):
        self.generator.eval()
        self.discriminator.eval()
        result = {key: 0 for key in ['adv_g', 'fm', 'loss_D', 'ms_mel_loss', 'commitment_loss', 'codebook_loss', 'LSD_L', 'LSD_H', 'subband_loss', 'fullband_loss']}
        
        with torch.no_grad():
            for i, (hr, lr, cond, _, sfm) in enumerate(tqdm(self.val_loader, desc='Validation')):
                lr, hr, cond, sfm = lr.to(self.device), hr.to(self.device), cond.to(self.device), sfm.to(self.device)   

                # analysis
                input_subbands, target_subbands = obtain_subbands(self.config, self.pqmf_fb, lr, hr)

                if self.use_sfm:
                    hf_estimate, commitment_loss, codebook_loss = self._forward_pass(input_subbands, cond, sfm, hr=hr)
                else:
                    hf_estimate, commitment_loss, codebook_loss = self._forward_pass(input_subbands, cond, hr=hr)
                    
                # synthesis
                hr_target, x_hat_full = synthesize_subbands(hf_estimate, target_subbands, input_subbands, self.pqmf_fb, hr.shape[-1])
        
                loss_G, ms_mel_loss_value, g_loss_dict, g_loss_report, subband_loss_value, fullband_loss_value = self.loss_calculator.compute_generator_loss(
                                                                                                hr_target, x_hat_full, commitment_loss, codebook_loss,
                                                                                                hf_estimate=hf_estimate, target_subbands=target_subbands)  
                loss_D, d_loss_dict, d_loss_report = self.loss_calculator.compute_discriminator_loss(hr_target, x_hat_full)

                # Compute LSD and LSD_H metrics
                cutoff_freq = self.config['generator']['c_in'] * 24000 / 32

                batch_lsd_l = lsd_batch(x_batch=hr_target.cpu(), y_batch=x_hat_full.cpu(), fs=48000, start=0, cutoff_freq=cutoff_freq)
                batch_lsd_h = lsd_batch(x_batch=hr_target.cpu(), y_batch=x_hat_full.cpu(), fs=48000, start=cutoff_freq, cutoff_freq=24000)
                result['LSD_L'] += batch_lsd_l
                result['LSD_H'] += batch_lsd_h

                # Aggregate results
                result['adv_g'] += g_loss_dict.get('adv_g', 0).item()
                result['fm'] += g_loss_dict.get('fm', 0).item()
                result['ms_mel_loss'] += ms_mel_loss_value.item() if ms_mel_loss_value else 0
                result['loss_D'] += loss_D.item()
                result['commitment_loss'] += commitment_loss.item() if commitment_loss else 0
                result['codebook_loss'] += codebook_loss.item() if codebook_loss else 0
                result['subband_loss'] += subband_loss_value.item() if subband_loss_value else 0
                result['fullband_loss'] += fullband_loss_value.item() if fullband_loss_value else 0    
                # import pdb
                # pdb.set_trace()
                
                # Data logging
                if i in [0,5,33]:  
                    if not self.hr_logged:
                        self.unified_log({
                            f'audio_hr_{i}': hr_target.squeeze().cpu().numpy(),
                            f'spec_hr_{i}': draw_spec(hr_target.squeeze().cpu().numpy(),win_len=2048, sr=48000, use_colorbar=False, hop_len=1024, return_fig=True),
                            # f'spec_lr_{i}': draw_spec(lr.squeeze().cpu().numpy(),win_len=2048, sr=48000, use_colorbar=False, hop_len=1024, return_fig=True),
                        }, 'val', step=step)

                    self.unified_log({
                        f'audio_bwe_{i}': x_hat_full.squeeze().cpu().numpy(),
                        f'spec_bwe_{i}': draw_spec(x_hat_full.squeeze().cpu().numpy(),win_len=2048, sr=48000, use_colorbar=False, hop_len=1024, return_fig=True),
                    }, 'val', step=step)
            self.hr_logged = True

        for key in result:
            result[key] /= len(self.val_loader)
        return result
        
    def load_checkpoints(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
            self.optim_D.load_state_dict(checkpoint['optim_D_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"Checkpoint loaded successfully from {checkpoint_path} at epoch {self.start_epoch}.")
            # start_step
            if 'step' in checkpoint:
                self.start_step = checkpoint['step']
                print(f"Resuming from step {self.start_step}.")
        else:
            raise ValueError(f"No checkpoint found at {checkpoint_path}.")


    def save_checkpoint(self, epoch, val_result, save_path, step=None):
        os.makedirs(save_path, exist_ok=True)
        if step is not None:
            filename = f"step_{step/1000:.1f}k_lsdh_{val_result['LSD_H']:.4f}.pth"
        else:
            filename = f"epoch_{epoch}_lsdh_{val_result['LSD_H']:.4f}.pth"
        save_path = os.path.join(save_path, filename)
        
        torch.save({
            'epoch': epoch,
            'step' : step, 
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optim_G_state_dict': self.optim_G.state_dict(),
            'optim_D_state_dict': self.optim_D.state_dict(),
        }, save_path)

    def train(self, num_epochs):
        if self.config['train']['val_step']:
            val_step = self.config['train']['val_step']
            
        best_lsdh = float('inf')
        if self.start_step > 1:
            global_step = self.start_step
            print(f"Resuming from step {self.start_step}.")
        else:
            global_step = (self.start_epoch-1) * len(self.train_loader)
                
        for epoch in range(self.start_epoch,num_epochs+1):
            train_result={}
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{num_epochs}')

            for hr, lr, cond, name, sfm in progress_bar:
                global_step += 1
                hr, lr, cond, sfm= hr.to(self.device), lr.to(self.device), cond.to(self.device), sfm.to(self.device)
                step_result = self.train_step(hr, lr, cond, step=global_step,  sfm=sfm,  pretrain_step=self.config['train']['pretrain_step'])

                # Sum up step losses for logging purposes
                for key, value in step_result.items():
                    train_result[key] = train_result.get(key, 0) + value

                # Update tqdm description with specific losses
                progress_bar.set_postfix({
                    'loss_G': step_result.get('loss_G', 0),
                    'mel_loss': step_result.get('ms_mel_loss', 0),
                    'subband_loss': step_result.get('subband_loss', 0),
                })
                # update scheduler
                self.scheduler_G.step()
                self.scheduler_D.step()
                
                # validation
                if global_step % val_step == 0:
                    val_result = self.validate(global_step)
                    self.unified_log(val_result, 'val', step=global_step)
                    if val_result['LSD_H'] < best_lsdh:
                        # best_lsdh = val_result['LSD_H']
                        print(f"Ckpt saved at {self.config['train']['ckpt_save_dir']} with LSDH {val_result['LSD_H']:.4f}")
                        self.save_checkpoint(epoch, val_result, save_path=self.config['train']['ckpt_save_dir'], step=global_step)
                        
def obtain_subbands(config, pqmf_fb, lr, hr):
    """Perform PQMF analysis to extract input and target subbands."""
    input_subbands = pqmf_fb.analysis(lr)[:, :config['generator']['c_in'], :]  # Core bands [B,5,T]
    target_subbands = pqmf_fb.analysis(hr)[:, config['generator']['c_in']:config['generator']['c_in'] + config['generator']['c_out'], :]  # High bands [B,27,T]
    return input_subbands, target_subbands


def synthesize_subbands(hf_estimate, target_subbands, input_subbands, pqmf_fb, target_length):
    """Reconstruct fullband waveforms using PQMF synthesis."""
    _b, _f1, _l = input_subbands.shape
    _b, _f2, _l = target_subbands.shape
    freq_zeros = torch.zeros(_b,32-_f1-_f2,_l).to(input_subbands.device)  # Zero padding

    x_hat = torch.cat((input_subbands.detach(), hf_estimate, freq_zeros), dim=1)
    x_hat_full = pqmf_fb.synthesis(x_hat, delay=0, length=target_length)

    hr_target = pqmf_fb.synthesis(torch.cat((input_subbands, target_subbands, freq_zeros), dim=1), delay=0, length=target_length)
    return hr_target, x_hat_full

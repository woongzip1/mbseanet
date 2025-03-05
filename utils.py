from matplotlib import pyplot as plt
import numpy as np
import librosa
import os
import torch
import wandb
from scipy.signal import stft

def count_model_params(model):
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    
def plot_signals(x, x_hat, range=[10000,15000],FIGSIZE=(10,2), diff=0.01):
    plt.figure(figsize=FIGSIZE)
    plt.plot(x, label='gt')
    plt.plot(x_hat+diff, label='s')
    plt.xlim(range)
    plt.legend()
    plt.show()
    
def draw_spec(x,
              figsize=(10, 6), title='', n_fft=2048,
              win_len=1024, hop_len=256, sr=16000, cmap='inferno',
              vmin=-50, vmax=40, use_colorbar=True,
              ylim=None,
              title_fontsize=10,
              label_fontsize=8,
                return_fig=False,
                save_fig=False, save_path=None):
    fig = plt.figure(figsize=figsize)
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    stft = 20 * np.log10(np.clip(np.abs(stft), a_min=1e-8, a_max=None))

    r=5
    # stft[...,100-r:100+r] = -50
    
    plt.imshow(stft,
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
               origin='lower', extent=[0, len(x) / sr, 0, sr//2])

    if use_colorbar:
        plt.colorbar()

    plt.xlabel('Time (s)', fontsize=label_fontsize)
    plt.ylabel('Frequency (Hz)', fontsize=label_fontsize)

    if ylim is None:
        ylim = (0, sr / 2)
    plt.ylim(*ylim)

    plt.title(title, fontsize=title_fontsize)
    
    if save_fig and save_path:
        plt.savefig(f"{save_path}.png")
    
    if return_fig:
        plt.close()
        return fig
    else:
        # plt.close()
        plt.show()
        return stft
    
def plot_magnitude(h, db=False, returnmag=False, figsize=(12,3)):
    """
    Plots the magnitude and phase response of a filter given its coefficients.

    Parameters:
        h (array-like): Filter coefficients.
    """
    # Frequency range from 0 to pi
    omega = np.linspace(0, np.pi, 500)

    # Compute DTFT
    H = np.array([np.sum(h * np.exp(-1j * w * np.arange(len(h)))) for w in omega])

    # Magnitude and phase response
    magnitude = np.abs(H)
    if db: magnitude = 20 * np.log10(magnitude)
    phase = np.angle(H)
    # Plotting
    plt.figure(figsize=figsize)
    # Magnitude response
    plt.plot(omega, magnitude)
    plt.title("Magnitude Response")
    plt.xlabel("Frequency (rad/sample)")
    plt.ylabel("|H(ω)|")
    if db: plt.ylabel("|H(ω)| db")
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    if db: plt.ylim([-100, 50])
    plt.grid()
    
    if returnmag:
        return magnitude
    

def plot_response(h, db=False, returnmag=False):
    """
    Plots the magnitude and phase response of a filter given its coefficients.

    Parameters:
        h (array-like): Filter coefficients.
    """
    # Frequency range from 0 to pi
    omega = np.linspace(0, np.pi, 500)

    # Compute DTFT
    H = np.array([np.sum(h * np.exp(-1j * w * np.arange(len(h)))) for w in omega])

    # Magnitude and phase response
    magnitude = np.abs(H)
    if db: magnitude = 20 * np.log10(magnitude)
    phase = np.angle(H)
    # Plotting
    plt.figure(figsize=(12, 6))

    # Magnitude response
    plt.subplot(2, 1, 1)
    plt.plot(omega, magnitude)
    plt.title("Magnitude Response")
    plt.xlabel("Frequency (rad/sample)")
    plt.ylabel("|H(ω)|")
    if db: plt.ylabel("|H(ω)| db")
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    if db: plt.ylim([-100, 50])
    plt.grid()

    # Phase response
    plt.subplot(2, 1, 2)
    plt.plot(omega, phase)
    plt.title("Phase Response")
    plt.xlabel("Frequency (rad/sample)")
    plt.ylabel("Phase (radians)")
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    plt.grid()

    plt.tight_layout()
    plt.show()
    
    if returnmag:
        return magnitude
    
""" Get list of all audio paths """
def get_audio_paths(paths: list, file_extensions=['.wav', '.flac']):
    audio_paths = []
    if isinstance(paths, str):
        paths = [paths]
        
    for path in paths:
        for root, dirs, files in os.walk(path):
            audio_paths += [os.path.join(root, file) for file in files if os.path.splitext(file)[-1].lower() in file_extensions]
                        
    audio_paths.sort(key=lambda x: os.path.split(x)[-1])
    
    return audio_paths

def get_filename(path):
    return os.path.splitext(os.path.basename(path))


# def wandb_log(loglist, epoch, note):
#     for key, val in loglist.items():
#         if isinstance(val, torch.Tensor):
#             item = val.cpu().detach().numpy()
#         else:
#             item = val
#         try:
#             if isinstance(item, float):
#                 log = item
#             elif isinstance(item, plt.Figure):
#                 log = wandb.Image(item)
#                 plt.close(item)
#             elif item.ndim in [2, 3]:  # 이미지 데이터
#                 log = wandb.Image(item, caption=f"{note.capitalize()} {key.capitalize()} Epoch {epoch}")
#             elif item.ndim == 1:  # 오디오 데이터
#                 log = wandb.Audio(item, sample_rate=48000, caption=f"{note.capitalize()} {key.capitalize()} Epoch {epoch}")
#             else:
#                 log = item
#         except Exception as e:
#             print(f"Failed to log {key}: {e}")
#             log = item

#         wandb.log({
#             f"{note.capitalize()} {key.capitalize()}": log,
#         }, step=epoch)

def lsd_batch(x_batch, y_batch, fs=16000, frame_size=0.02, frame_shift=0.02, start=0, cutoff_freq=0, nfft=512):
    frame_length = int(frame_size * fs)
    frame_step = int(frame_shift * fs)

    if fs == 48000:
        frame_length = 2048
        frame_step = 2048
        nfft = 2048

    if isinstance(x_batch, np.ndarray):
        x_batch = torch.from_numpy(x_batch)
        y_batch = torch.from_numpy(y_batch)
   
    if x_batch.dim()==1:
        batch_size = 1
    ## 1 x 32000
    elif x_batch.dim()==2:
        x_batch=x_batch.unsqueeze(1)
    batch_size, _, signal_length = x_batch.shape
   
    if y_batch.dim()==1:
        y_batch=y_batch.reshape(batch_size,1,-1)
    elif y_batch.dim()==2:
        y_batch=y_batch.unsqueeze(1)
   
    # X and Y Size
    x_len = x_batch.shape[-1]
    y_len = y_batch.shape[-1]
    minlen = min(x_len, y_len)
    x_batch = x_batch[:,:,:minlen]
    y_batch = y_batch[:,:,:minlen]

    lsd_values = []
    for i in range(batch_size):
        x = x_batch[i, 0, :].numpy()
        y = y_batch[i, 0, :].numpy()
 
        # STFT
        ## nfft//2 +1: freq len
        f_x, t_x, Zxx_x = stft(x, fs, nperseg=frame_length, noverlap=frame_length - frame_step, nfft=nfft)
        f_y, t_y, Zxx_y = stft(y, fs, nperseg=frame_length, noverlap=frame_length - frame_step, nfft=nfft)
       
        # Power spec
        power_spec_x = np.abs(Zxx_x) ** 2
        power_spec_y = np.abs(Zxx_y) ** 2
       
        # Log Power Spec
        log_spec_x = np.log10(power_spec_x + 1e-10)  # eps
        log_spec_y = np.log10(power_spec_y + 1e-10)

        if start or cutoff_freq:
            freq_len = log_spec_x.shape[0]
            max_freq = fs // 2
            start = int(start / max_freq * freq_len)
            freq_idx = int(cutoff_freq / max_freq * freq_len)
            log_spec_x = log_spec_x[start:freq_idx,:]
            log_spec_y = log_spec_y[start:freq_idx,:]

        #Spectral Mean
        lsd = np.sqrt(np.mean((log_spec_x - log_spec_y) ** 2, axis=0))
       
        #Frame mean
        mean_lsd = np.mean(lsd)
        lsd_values.append(mean_lsd)
   
    # Batch mean
    batch_mean_lsd = np.mean(lsd_values)
    # return log_spec_x, log_spec_y
    return batch_mean_lsd

def print_config(config, indent=0):
    for k, v in config.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_config(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")


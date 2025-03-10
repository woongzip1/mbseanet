from matplotlib import pyplot as plt
import torchaudio as ta
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torchaudio.compliance.kaldi as kaldi
from utils import *

class CustomDataset(Dataset):
    """ PATH = [dir1, dir2 , ...] 

    path_dir_wb=[ "/mnt/hdd/Dataset/FSD50K_16kHz", 
                "/mnt/hdd/Dataset/MUSDB18_HQ_16kHz_mono"],
    path_dir_nb=["/mnt/hdd/Dataset/FSD50K_16kHz_codec",
                 "/mnt/hdd/Dataset/MUSDB18_MP3_8k"],
                 """
    def __init__(self, path_dir_nb, path_dir_wb, seg_len=0.9, sr=48000, mode="train", 
                 start_index=5, high_index=31, use_sfm=False, use_pqmf=False):
        assert isinstance(path_dir_nb, list), "PATH must be a list"

        self.seg_len = seg_len
        self.mode = mode
        self.sr = sr
        self.high_index = high_index
        self.start_index = start_index
        self.use_sfm = use_sfm
        self.use_pqmf = use_pqmf
        
        paths_wav_wb = []
        paths_wav_nb = []
        self.labels = []
        self.path_lengths = {}
    
        # number of dataset -> ['path1','path2']
        for i in range(len(path_dir_nb)):
            self.path_dir_nb = path_dir_nb[i]
            self.path_dir_wb = path_dir_wb[i]

            wb_files = get_audio_paths(self.path_dir_wb, file_extensions='.wav')
            nb_files = get_audio_paths(self.path_dir_nb, file_extensions='.wav')
            paths_wav_wb.extend(wb_files)
            paths_wav_nb.extend(nb_files)

            # Assign labels based on path1, path2
            self.labels.extend([i] * len(wb_files)) 
            self.path_lengths[f'idx{i}len'] = len(wb_files)
            print(f"Index:{i} with {len(wb_files)} samples")

        if len(paths_wav_wb) != len(paths_wav_nb):
            raise ValueError(f"Error: LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers are different!")

        # make filename wb-nb        
        self.filenames = list(zip(paths_wav_wb, paths_wav_nb))
        # print(f"{mode}: {len(self.filenames)} files loaded")
        print(f"LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers loaded!")


    def get_class_counts(self):
        return [self.path_lengths[f'idx{i}len'] for i in range(len(self.path_lengths))]

    def _multiple_pad(self, wav, N=2048):
        pad_len = (N - wav.shape[-1] % N) % N
        wav = torch.nn.functional.pad(wav, (0, pad_len), mode='constant', value=0)
        return wav

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        N = 2048
        path_wav_wb, path_wav_nb = self.filenames[idx]
        # get label with current nb-wb pair
        label = self.labels[idx]

        wav_nb, sr_nb = ta.load(path_wav_nb)
        wav_wb, sr_wb = ta.load(path_wav_wb)

        wav_wb = wav_wb.view(1, -1)
        wav_nb = wav_nb.view(1, -1)

        if self.seg_len > 0 and self.mode == "train": # train mode
            duration = int(self.seg_len * self.sr) # 43200
            duration = (duration // N) * N # multiple of N
            
            if wav_nb.shape[-1] < duration:
                wav_nb = self.ensure_length(wav_nb, duration)
                wav_wb = self.ensure_length(wav_wb, duration)
            elif wav_nb.shape[-1] > duration:
                start_idx = np.random.randint(0, wav_nb.shape[-1] - duration)
                wav_nb = wav_nb[:, start_idx:start_idx + duration]
                wav_wb = wav_wb[:, start_idx:start_idx + duration]

        elif self.mode == "val": 
            wav_nb = self._multiple_pad(wav_nb)
            wav_wb = self._multiple_pad(wav_wb)            
            # wav_nb = self.ensure_length(wav_nb, int(self.seg_len * self.sr))
            # wav_wb = self.ensure_length(wav_wb, int(self.seg_len * self.sr))
        else:
            sys.exit(f"unsupported mode! (train/val)")

        if self.use_pqmf:
            return wav_wb, wav_nb, 0, get_filename(path_wav_wb)[0], label

        if self.use_sfm:
            spec = self.get_spectrogram(wav_wb, power=1.0, log_scale=False)
            spec = self.extract_subband(spec, start=self.start_index, end=self.high_index)
            sfm = self.extract_sfm(spec)        # sfm
            spec = spec.pow(2.0)
            spec = ta.transforms.AmplitudeToDB()(spec)
            spec = self.normalize_spec(spec)    # normalized power spec
            return wav_wb, wav_nb, spec, get_filename(path_wav_wb)[0], sfm
        else:
            spec = self.get_spectrogram(wav_wb, power=2.0, log_scale=True) 
            spec = self.normalize_spec(spec)
            spec = self.extract_subband(spec, start=self.start_index, end=self.high_index) # start:5 : 3750Hz

        return wav_wb, wav_nb, spec, get_filename(path_wav_wb)[0], label

    @staticmethod
    def ensure_length(wav, target_length):
        target_length = int(target_length)
        if wav.shape[1] < target_length:
            pad_size = target_length - wav.shape[1]
            wav = F.pad(wav, (0, pad_size))
        elif wav.shape[1] > target_length:
            wav = wav[:, :target_length]
        return wav
        
    def set_maxlen(self, wav, max_lensec):
        sr = self.sr
        max_len = int(max_lensec * sr)
        if wav.shape[1] > max_len:
            # print(wav.shape, max_len)
            wav = wav[:, :max_len]
        return wav

    @staticmethod
    def get_spectrogram(waveform, N=2048, power=2.0, log_scale=True):
        n_fft = 2048
        hop_length = 2048 
        win_length = 2048

        # pad at the end to make (mod N)
        padlen = (N - waveform.shape[-1] % N) % N
        waveform = F.pad(waveform, (0, padlen))

        spectrogram = ta.transforms.Spectrogram(
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            power=power,
            center=False,
        )(waveform)

        if log_scale:
            spectrogram = ta.transforms.AmplitudeToDB()(spectrogram)
            
        return spectrogram  

    def normalize_spec(self, spec):
        norm_mean = -42.61
        norm_std = 25.79
        spec = (spec - norm_mean) / (norm_std * 2)
        return spec
    
    def extract_subband(self, spec, start=5, end=31):
        """ Get spectrogram Inputs and extract range of subbands : [start:end] 
        start: start index of subband (from 0)
        start=0: from first subband
        start=N: from N+1 th subband
        Total 32 subbands
        """
        
        C,F,T = spec.shape # C F T
        num_subband = 32
        freqbin_size = F // num_subband # 1024//32

        dc_line = spec[:,0,:].unsqueeze(1)
        
        f_start = 1 + freqbin_size * start
        f_end = 1 + freqbin_size * (end+1)
        
        extracted_spec = spec[:,f_start:f_end,:]
        if f_start == 1:
            extracted_spec = torch.cat((dc_line, extracted_spec),dim=1) # [C,F,T]

        # print(f_start/1024 * 24000, f_end/1024 * 24000)
        return extracted_spec # C, F, T

    def extract_sfm(self, S, eps=1e-8):
        C,F,T = S.shape # unbatched input
        S = torch.abs(S)
        num_bands = F // 32  
        S = S.view(C, num_bands, 32, -1)  # [C,F,T] → [C,F//32,32,T]
        
        geometric_mean = torch.exp(torch.mean(torch.log(S + eps), dim=-2)) # Freq-axis pooling
        arithmetic_mean = torch.mean(S, dim=-2)  
        
        sfm_bands = geometric_mean / (arithmetic_mean + eps) 
        sfm_bands = 10 * torch.log10(sfm_bands + eps).squeeze(1)
        
        return sfm_bands.squeeze(0)  # Shape: (F//32,T)
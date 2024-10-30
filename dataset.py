import os
import pickle
import numpy as np 
import imageio
import scipy.signal as sig
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import rf.organizer as org
from rf.proc import create_fast_slow_matrix, find_range, rotateIQ


class FusionDataset(Dataset):
    def __init__(self, rgb_datapath, rf_datapath, video_list, rf_file_list, recording_str="rgbd_rgb", ppg_str="rgbd",
                 file_length=900, frame_length=128, sampling_ratio=4, samples=256, train=False,
                 ppg_offset=25, num_samps=30,
                 samp_f=5e6, freq_slope=60.012e12, window_size=5) -> None:
        # For RF&RGB
        self.ppg_offset = ppg_offset # There is an offset in capturing the signals w.r.t the ground truth.
        self.num_samps = num_samps # Number of samples to be created by oversampling one trial.
        self.file_length = file_length # Number of frames in the input video. (Requires all data-samples to have the same number of frames).
        self.frame_length = frame_length # Number of frames in the output tensor sample.
        self.rgb_datapath = rgb_datapath # Data structure for videos
        self.rf_datapath = rf_datapath
        self.train = train
        
        # For RGB only
        self.id_str = recording_str  # Name of the files being read.
        self.ppg_str = ppg_str  # Name of the files being read.
        self.video_list = video_list# Load videos and signals.
        
        # For RF only
        self.rf_file_list = rf_file_list
        self.sampling_ratio = sampling_ratio
        self.samples=samples
        self.samp_f = samp_f
        self.freq_slope = freq_slope
        self.window_size = window_size

        # 读取真值信号
        self.signal_list = []
        remove_folders = []
        for folder in self.video_list:
            file_path = os.path.join(self.rgb_datapath, folder)
            # Make a list of the folder that do not have the PPG signal.
            if(os.path.exists(file_path)):
                if(os.path.exists(os.path.join(file_path, f"{self.ppg_str}_ppg.npy"))):
                    signal = np.load(os.path.join(file_path, f"{self.ppg_str}_ppg.npy"))
                    self.signal_list.append(signal[self.ppg_offset:])#ppg信号列表
                else:
                    print(folder, "ppg doesn't exist.")
                    remove_folders.append(folder)
            else:
                print(folder, " doesn't exist.")
                remove_folders.append(folder)
        # Remove the unuseless ppg signal
        for i in remove_folders:
            self.video_list.remove(i)    
            print("Removed", i)

        # 对真值信号进行正则化
        self.signal_list = np.array(self.signal_list)
        self.vital_mean = np.mean(self.signal_list)
        self.vital_std = np.std(self.signal_list)
        self.signal_list = (self.signal_list - self.vital_mean)/self.vital_std

        # 创建索引
        self.video_nums = np.arange(0, len(self.video_list))
        self.all_idxs = []
        for num in self.video_nums:
            # Generate the start index.随机生成30个起始索引
            rgb_cur_frame_nums = np.random.randint(low=0, high =self.file_length - self.frame_length - self.ppg_offset, size = self.num_samps)
            # Append all the start indices.
            rf_cur_frame_nums = rgb_cur_frame_nums * self.sampling_ratio
            for rgb_cur_frame_num, rf_frame_num in zip(rgb_cur_frame_nums, rf_cur_frame_nums):
                self.all_idxs.append((num, (rgb_cur_frame_num,rf_frame_num)))

        # 预处理RF数据
        self.rf_data_list = []
        for rf_file in self.rf_file_list:
            # Read the raw RF data
            rf_fptr = open(os.path.join(self.rf_datapath, rf_file, "rf.pkl"), 'rb')
            s = pickle.load(rf_fptr)
            # Organize the raw data from the RF.# Number of samples is set to 128 for our experiments.
            rf_organizer = org.Organizer(s, 1, 1, 1, 2 * self.samples)
            frames = rf_organizer.organize()
            # The RF read adds zero alternatively to the samples. Remove these zeros.
            frames = frames[:, :, :, 0::2]  # (T, 1, 1, F)
            # 创建两通道的快慢矩阵
            data_f = create_fast_slow_matrix(frames) # (T, F)
            
            # 下面方法可加可不加
            range_index = find_range(data_f, self.samp_f, self.freq_slope, self.samples)  # TODO
            data_f = data_f[:, range_index - self.window_size // 2:range_index + self.window_size // 2 + 1]  # (T, window_size)
            
            self.rf_data_list.append(data_f)

    def __len__(self):
        return int(len(self.all_idxs))

    def __getitem__(self, idx):
        """"根据索引idx返回 np.array(rf_item): (C, T, F) ,np.array(rgb_item): (C, T, H, W), np.array(item_sig): (T, ) """
        file_num, (rgb_frame_start, rf_frame_start) = self.all_idxs[idx]
        # 获取RGB图像数据
        rgb_item = []
        for img_idx in range(self.frame_length):
            image_path = os.path.join(self.rgb_datapath, 
                                str(self.video_list[file_num]),
                                f"{self.id_str}_{rgb_frame_start+img_idx}.png")
            rgb_item.append(imageio.imread(image_path))
        rgb_item = np.array(rgb_item)
        # 处理RGB数据
        if (len(rgb_item.shape) < 4):
            rgb_item = np.expand_dims(rgb_item, axis=3)
        rgb_item = np.transpose(rgb_item, axes=(3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)。
        if (rgb_item.dtype == np.uint16):
            rgb_item = rgb_item.astype(np.int32)

        # 获取RF数据
        data_f = self.rf_data_list[file_num]
        rf_item = data_f[rf_frame_start: rf_frame_start + (self.sampling_ratio * self.frame_length), :] # (T, F)
        # 生成RJTF图
        if self.train and torch.rand(1) < 0.5:
            rf_item = rotateIQ(rf_item)
        rf_item= self.get_rjtf(rf_item) # (C, T, F)

        # 获取Sig信号.
        item_sig = self.signal_list[file_num][rgb_frame_start:rgb_frame_start + self.frame_length]
        assert len(item_sig) == self.frame_length, f"Expected signal of length {self.frame_length}, but got signal of length {len(item_sig)}"

        return rf_item, np.array(rgb_item), np.array(item_sig)

    def get_rjtf(self, data_f):
        sig = torch.from_numpy(data_f).sum(-1)
        win = torch.hamming_window(36)
        # TODO: log magnitude of both \alpha and \theta ? or complex input
        sig_mag = torch.abs(sig) * 1 + 0j
        rjtf_mag = torch.stft(sig_mag, 256, 8, win_length=36, window=win, return_complex=False).permute(2, 1, 0)
        rjtf_mag = (rjtf_mag - rjtf_mag.mean()) / rjtf_mag.std()  # 2, T, F
        
        sig_phase = torch.exp(1j * torch.angle(sig))
        rjtf_phase = torch.stft(sig_phase, 256, 8, win_length=36, window=win, return_complex=False).permute(2, 1, 0)
        rjtf_phase = (rjtf_phase - rjtf_phase.mean()) / rjtf_phase.std()  # 2, T, F
        
        # # 可要可不要
        # rjtf_mag = torch.view_as_complex(rjtf_mag.permute(1, 2, 0)) # T, F, 2
        # rjtf_mag = torch.log(torch.abs(rjtf_mag) + 1e-8) # T, F
        # rjtf_phase = torch.view_as_complex(rjtf_phase.permute(1, 2, 0)) # T, F, 2
        # rjtf_phase = torch.log(torch.abs(rjtf_phase) + 1e-8) # T, F
        
        rjtf = torch.cat([rjtf_mag, rjtf_phase])  
        rjtf = F.interpolate(rjtf.unsqueeze(0), size=(self.frame_length, 256), mode="bicubic", align_corners=True)[0]
        return rjtf   # 4，128，256 或 2，128，256

class RGBData(Dataset):
    def __init__(self, datapath, video_list, recording_str="rgbd_rgb", ppg_str="rgbd",
                 video_length = 900, frame_length = 64) -> None:
        
        # There is an offset in capturing the signals w.r.t the ground truth.
        self.ppg_offset = 25
        # Number of samples to be created by oversampling one trial.
        self.num_samps = 30
        # Name of the files being read. Name depends on how the file was save. We have saved the file as rgbd_rgb
        self.id_str = recording_str
        self.ppg_str = ppg_str
        # Number of frames in the input video. (Requires all data-samples to have the same number of frames).
        self.video_length = video_length
        # Number of frames in the output tensor sample.
        self.frame_length = frame_length
        
        # Data structure for videos.
        self.datapath = datapath
        # Load videos and signals.
        self.video_list = video_list
        # The PPG files for the RGB are stored as rgbd_ppg and not rgbd_rgb_ppg.

        self.signal_list = []
        # Load signals
        remove_folders = []
        # print("datapath: ", datapath)
        for folder in self.video_list:
            # print("folder: ", folder)
            file_path = os.path.join(datapath, folder)
            # Make a list of the folder that do not have the PPG signal.
            if(os.path.exists(file_path)):
                if(os.path.exists(os.path.join(file_path, f"{self.ppg_str}_ppg.npy"))):
                    signal = np.load(os.path.join(file_path, f"{self.ppg_str}_ppg.npy"))
                    self.signal_list.append(signal[self.ppg_offset:])
                else:
                    print(folder, "ppg doesn't exist.")
                    remove_folders.append(folder)
            else:
                print(folder, " doesn't exist.")
                remove_folders.append(folder)
        # Remove the PPGs
        for i in remove_folders:
            self.video_list.remove(i)    
            print("Removed", i)

        # Extract the stats for the vital signs.
        self.signal_list = np.array(self.signal_list)
        self.vital_mean = np.mean(self.signal_list)
        self.vital_std = np.std(self.signal_list)
        self.signal_list = (self.signal_list - self.vital_mean)/self.vital_std
        # print(f"真值信号数量{len(self.signal_list)}\t每个真值信号的形状：{self.signal_list[0].shape}")

        # Create a list of video number and valid frame number to extract the data from.
        self.video_nums = np.arange(0, len(self.video_list))
        self.frame_nums = np.arange(0, self.video_length - frame_length - self.ppg_offset)
        
        # Create all possible sampling combinations.
        self.all_idxs = []
        for num in self.video_nums:
            # Generate the start index.
            cur_frame_nums = np.random.randint(low=0, 
                                               high = self.video_length - frame_length - self.ppg_offset, 
                                               size = self.num_samps)
            # Append all the start indices.
            for cur_frame_num in cur_frame_nums:
                self.all_idxs.append((num,cur_frame_num))
            
            
    def __len__(self):
        return int(len(self.all_idxs))
    def __getitem__(self, idx):
        # Get the video number and the starting frame index.
        video_number, frame_start = self.all_idxs[idx]
        # Get video frames for the output video tensor.
        # (Expects each sample to be stored in a folder with the sample name. Each frame is stored as a png)
        item = []
        for img_idx in range(self.frame_length):
            image_path = os.path.join(self.datapath, 
                                str(self.video_list[video_number]), 
                                f"{self.id_str}_{frame_start+img_idx}.png")
            item.append(imageio.imread(image_path))
        item = np.array(item)

        # Add channel dim if no channels in image.
        if(len(item.shape) < 4): 
            item = np.expand_dims(item, axis=3)
        item = np.transpose(item, axes=(3,0,1,2))
        # Get signal.
        item_sig = self.signal_list[int(video_number)][int(frame_start):int(frame_start+self.frame_length)]
        
        # Patch for the torch constructor. uint16 is a not an acceptable data-type.
        if(item.dtype == np.uint16):
            item = item.astype(np.int32)
        return np.array(item), np.array(item_sig)






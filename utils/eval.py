import os
import pickle
import numpy as np
import scipy.stats
import sklearn.metrics
import torch
import sys
from tqdm import tqdm
import torch.nn.functional as F
from .errors import getErrors, cal_mae_rmse_r
from .utils import extract_video, pulse_rate_from_power_spectral_density
# sys.path.append("..")
from rf import organizer as org
from rf.proc import create_fast_slow_matrix, find_range


# from model import FusionModel

def eval_model(model, video_list, rf_file_list, rgb_datapath="/share2/data/zhouwenqing/UCLA-rPPG/rgb_files",
               rf_datapath="/share2/data/zhouwenqing/UCLA-rPPG/rf_files", file_name="rgbd_rgb",
               ppg_file_name="rgbd_ppg.npy",
               frame_length=128, sampling_ratio=4, samples=256, ppg_offset=25,
               samp_f=5e6, freq_slope=60.012e12, window_size=5, device=torch.device('cpu')):
    model.eval()
    video_samples = []
    assert len(video_list) == len(rf_file_list)
    num_vals = len(video_list)

    for rgb_file, rf_file in tqdm(zip(video_list, rf_file_list), total=len(video_list)):
        # 一个字典，记录每个视频的est_ppg、gt_ppgs
        cur_video_sample = {}

        # 读取每个rgb视频和rf信号以及真值信号
        rgb_frames = extract_video(path=os.path.join(rgb_datapath, rgb_file), file_str=file_name)
        rf_fptr = open(os.path.join(rf_datapath, rf_file, "rf.pkl"), 'rb')
        target = np.load(os.path.join(rgb_datapath, rgb_file, ppg_file_name))

        # 预处理RF信号
        s = pickle.load(rf_fptr)
        rf_organizer = org.Organizer(s, 1, 1, 1, 2 * samples)
        rf_frames = rf_organizer.organize()
        rf_frames = rf_frames[:, :, :, 0::2]
        rf_frames = create_fast_slow_matrix(rf_frames)

        # 可加可不加
        range_index = find_range(rf_frames, samp_f, freq_slope, samples)
        temp_window = np.blackman(window_size)
        rf_frames = rf_frames[:, range_index-len(temp_window)//2:range_index+len(temp_window)//2 + 1]

        circ_buffer = rf_frames[0:800]
        rf_data = np.concatenate((rf_frames, circ_buffer))

        est_ppgs = None
        rgb_cut_index = np.arange(0, rgb_frames.shape[0] - ppg_offset, frame_length)
        # print(rgb_frames.shape[0], rgb_cut_index)
        rf_cut_index = rgb_cut_index * 4
        for cut_num in range(len(rgb_cut_index) - 1):
            # RGB
            rgb_cur_frames = rgb_frames[rgb_cut_index[cut_num]: rgb_cut_index[cut_num + 1], :, :, :]
            rgb_cur_frames = torch.from_numpy(rgb_cur_frames.astype(np.uint8)).permute(0, 3, 1, 2).float()
            rgb_cur_frames = rgb_cur_frames / 255
            rgb_cur_frames = rgb_cur_frames.to(device)

            # RF
            rf_cur_frames = rf_data[rf_cut_index[cut_num]: rf_cut_index[cut_num + 1], :]
            rf_cur_frames = get_rjtf(rf_cur_frames, frame_length, samples)
            rf_cur_frames = rf_cur_frames.to(device)

            # DL
            with torch.no_grad():
                # Add the B dim
                rgb_cur_frames = rgb_cur_frames.unsqueeze(0).float()
                rf_cur_frames = rf_cur_frames.unsqueeze(0).float()
                rgb_cur_frames = torch.transpose(rgb_cur_frames, 1, 2)
                # print(f"cut_num: {rgb_cut_index[cut_num]}, rgb_cur_frames: {rgb_cur_frames.shape}, rf_cur_frames: {rf_cur_frames.shape}")
                # Get the estimated PPG signal
                cur_est_ppg = model(rgb_cur_frames, rf_cur_frames)
                cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()
                # print(f"cut_num: {rgb_cut_index[cut_num]}, cur_est_ppg: {cur_est_ppg.shape}")
            # First sequence
            if est_ppgs is None:
                est_ppgs = cur_est_ppg
            else:
                est_ppgs = np.concatenate((est_ppgs, cur_est_ppg), -1)

        # Save
        cur_video_sample['est_ppgs'] = est_ppgs
        cur_video_sample['gt_ppgs'] = target[ppg_offset: ppg_offset + (len(rgb_cut_index) - 1) * frame_length]
        # print(f"cur_video_sample['est_ppgs']: {cur_video_sample['est_ppgs'].shape}, cur_video_sample['gt_ppgs']: {cur_video_sample['gt_ppgs'].shape}")
        video_samples.append(cur_video_sample)

    hr_window_size = 300
    stride = 128
    # stride = 1
    mae_list = []
    rmse_list = []
    pcc_list = []
    all_hr_est = []
    all_hr_gt = []
    for index, cur_video_sample in enumerate(video_samples):
        cur_est_ppgs = cur_video_sample['est_ppgs']

        # Load GT
        cur_gt_ppgs = cur_video_sample['gt_ppgs']
        cur_est_ppgs = (cur_est_ppgs - np.mean(cur_est_ppgs)) / np.std(cur_est_ppgs)
        cur_gt_ppgs = (cur_gt_ppgs - np.mean(cur_gt_ppgs)) / np.std(cur_gt_ppgs)

        # Get est HR for each window
        hr_est_temp = []
        hr_gt_temp = []
        for start in range(0, len(cur_est_ppgs) - hr_window_size, stride):
            ppg_est_window = cur_est_ppgs[start:start + hr_window_size]

            ppg_gt_window = cur_gt_ppgs[start:start + hr_window_size]

            # Normalize PPG
            ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
            ppg_gt_window = (ppg_gt_window - np.mean(ppg_gt_window)) / np.std(ppg_gt_window)
            hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            hr_gt_temp.append(pulse_rate_from_power_spectral_density(
                ppg_gt_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

        hr_est_windowed = np.array([hr_est_temp])
        hr_gt_windowed = np.array(hr_gt_temp)
        # print("hr_est_windowed.shape: ", hr_est_windowed.shape, "\thr_gt_windowed.shape: ", hr_gt_windowed.shape)
        # print(hr_est_temp == hr_gt_temp)
        all_hr_est.append(hr_est_temp)
        all_hr_gt.append(hr_gt_temp)

        # Errors
        RMSE, MAE, MAX, PCC = getErrors(hr_est_windowed, hr_gt_windowed)
        # print(index, MAE, RMSE, MAX, PCC)
        # MAE, RMSE, PCC = cal_mae_rmse_r(np.array(hr_est_temp), np.array(hr_gt_temp))

        mae_list.append(MAE)
        rmse_list.append(RMSE)
        pcc_list.append(PCC)
    # print('Mean MAE:', np.mean(np.array(mae_list)))
    return np.array(mae_list), np.array(rmse_list), np.array(pcc_list), (all_hr_est, all_hr_gt)


def eval_rgb_model(model, video_list, rgb_datapath="/share2/data/zhouwenqing/UCLA-rPPG/rgb_files",
                   file_name="rgbd_rgb", ppg_file_name="rgbd_ppg.npy",
                   frame_length=128, ppg_offset=25, device=torch.device('cpu')):
    model.eval()
    video_samples = []
    num_vals = len(video_list)

    for rgb_file in tqdm(video_list):
        # 一个字典，记录每个视频的est_ppg、gt_ppgs
        cur_video_sample = {}

        # 读取每个rgb视频以及真值信号
        rgb_frames = extract_video(path=os.path.join(rgb_datapath, rgb_file), file_str=file_name)
        target = np.load(os.path.join(rgb_datapath, rgb_file, ppg_file_name))

        est_ppgs = None
        rgb_cut_index = np.arange(0, rgb_frames.shape[0] - ppg_offset, frame_length)
        # print(rgb_frames.shape[0], rgb_cut_index)
        for cut_num in range(len(rgb_cut_index) - 1):
            # RGB
            rgb_cur_frames = rgb_frames[rgb_cut_index[cut_num]: rgb_cut_index[cut_num + 1], :, :, :]
            rgb_cur_frames = torch.from_numpy(rgb_cur_frames.astype(np.uint8)).permute(0, 3, 1, 2).float()
            rgb_cur_frames = rgb_cur_frames / 255
            rgb_cur_frames = rgb_cur_frames.to(device)

            # DL
            with torch.no_grad():
                # Add the B dim
                rgb_cur_frames = rgb_cur_frames.unsqueeze(0).float()
                rgb_cur_frames = torch.transpose(rgb_cur_frames, 1, 2)
                # print(f"cut_num: {rgb_cut_index[cut_num]}, rgb_cur_frames: {rgb_cur_frames.shape}, rf_cur_frames: {rf_cur_frames.shape}")
                # Get the estimated PPG signal
                cur_est_ppg = model(rgb_cur_frames)
                cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()
                # print(f"cut_num: {rgb_cut_index[cut_num]}, cur_est_ppg: {cur_est_ppg.shape}")
            # First sequence
            if est_ppgs is None:
                est_ppgs = cur_est_ppg
            else:
                est_ppgs = np.concatenate((est_ppgs, cur_est_ppg), -1)

        # Save
        cur_video_sample['est_ppgs'] = est_ppgs
        cur_video_sample['gt_ppgs'] = target[ppg_offset: ppg_offset + (len(rgb_cut_index) - 1) * frame_length]
        # print(f"cur_video_sample['est_ppgs']: {cur_video_sample['est_ppgs'].shape}, cur_video_sample['gt_ppgs']: {cur_video_sample['gt_ppgs'].shape}")
        video_samples.append(cur_video_sample)

    hr_window_size = 300
    stride = 128
    # stride = 1
    mae_list = []
    rmse_list = []
    pcc_list = []
    all_hr_est = []
    all_hr_gt = []
    for index, cur_video_sample in enumerate(video_samples):
        cur_est_ppgs = cur_video_sample['est_ppgs']

        # Load GT
        cur_gt_ppgs = cur_video_sample['gt_ppgs']
        cur_est_ppgs = (cur_est_ppgs - np.mean(cur_est_ppgs)) / np.std(cur_est_ppgs)
        cur_gt_ppgs = (cur_gt_ppgs - np.mean(cur_gt_ppgs)) / np.std(cur_gt_ppgs)

        # Get est HR for each window
        hr_est_temp = []
        hr_gt_temp = []
        for start in range(0, len(cur_est_ppgs) - hr_window_size, stride):
            ppg_est_window = cur_est_ppgs[start:start + hr_window_size]

            ppg_gt_window = cur_gt_ppgs[start:start + hr_window_size]

            # Normalize PPG
            ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
            ppg_gt_window = (ppg_gt_window - np.mean(ppg_gt_window)) / np.std(ppg_gt_window)
            hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            hr_gt_temp.append(pulse_rate_from_power_spectral_density(
                ppg_gt_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

        hr_est_windowed = np.array([hr_est_temp])
        hr_gt_windowed = np.array(hr_gt_temp)
        # print("hr_est_windowed.shape: ", hr_est_windowed.shape, "\thr_gt_windowed.shape: ", hr_gt_windowed.shape)
        # print(hr_est_temp == hr_gt_temp)
        all_hr_est.append(hr_est_temp)
        all_hr_gt.append(hr_gt_temp)

        # Errors
        RMSE, MAE, MAX, PCC = getErrors(hr_est_windowed, hr_gt_windowed)
        # MAE, RMSE, PCC = cal_mae_rmse_r(np.array(hr_est_temp), np.array(hr_gt_temp))

        mae_list.append(MAE)
        rmse_list.append(RMSE)
        pcc_list.append(PCC)
    # print('Mean MAE:', np.mean(np.array(mae_list)))
    return np.array(mae_list), np.array(rmse_list), np.array(pcc_list), (all_hr_est, all_hr_gt)


def eval_rf_model(model, rf_file_list, rf_datapath="/share2/data/zhouwenqing/UCLA-rPPG/rf_files",
                  frame_length=128, sampling_ratio=4, samples=256, ppg_offset=25,
                  samp_f=5e6, freq_slope=60.012e12, window_size=5, device=torch.device('cpu')):
    model.eval()
    video_samples = []
    for rf_file in tqdm(rf_file_list):
        # 一个字典，记录每个视频的est_ppg、gt_ppgs
        cur_video_sample = {}

        # 读取每个rf信号以及真值信号
        rf_fptr = open(os.path.join(rf_datapath, rf_file, "rf.pkl"), 'rb')
        target = np.load(f"{rf_datapath}/{rf_file}/vital_dict.npy", allow_pickle=True).item()['rgbd']['NOM_PLETHWaveExport']

        # 预处理RF信号
        s = pickle.load(rf_fptr)
        rf_organizer = org.Organizer(s, 1, 1, 1, 2 * samples)
        rf_frames = rf_organizer.organize()
        rf_frames = rf_frames[:, :, :, 0::2]
        rf_frames = create_fast_slow_matrix(rf_frames)

        # 可加可不加
        # range_index = find_range(rf_frames, samp_f, freq_slope, samples)
        # temp_window = np.blackman(window_size)
        # rf_frames = rf_frames[:, range_index-len(temp_window)//2:range_index+len(temp_window)//2 + 1]

        circ_buffer = rf_frames[0:800]
        rf_data = np.concatenate((rf_frames, circ_buffer))

        est_ppgs = None
        rf_cut_index = np.arange(0, rf_data.shape[0] - ppg_offset * sampling_ratio, frame_length * sampling_ratio)
        for cut_num in range(len(rf_cut_index) - 1):
            # RF
            rf_cur_frames = rf_data[rf_cut_index[cut_num]: rf_cut_index[cut_num + 1], :]
            rf_cur_frames = get_rjtf(rf_cur_frames, frame_length, samples)
            rf_cur_frames = rf_cur_frames.to(device)

            # DL
            with torch.no_grad():
                # Add the B dim
                rf_cur_frames = rf_cur_frames.unsqueeze(0).float()
                # Get the estimated PPG signal
                cur_est_ppg = model(rf_cur_frames)
                cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()
                # print(f"cut_num: {rgb_cut_index[cut_num]}, cur_est_ppg: {cur_est_ppg.shape}")
            # First sequence
            if est_ppgs is None:
                est_ppgs = cur_est_ppg
            else:
                est_ppgs = np.concatenate((est_ppgs, cur_est_ppg), -1)

        # Save
        cur_video_sample['est_ppgs'] = est_ppgs
        cur_video_sample['gt_ppgs'] = target[ppg_offset: ppg_offset + (len(rf_cut_index) - 1) * frame_length]
        # print(f"cur_video_sample['est_ppgs']: {cur_video_sample['est_ppgs'].shape}, cur_video_sample['gt_ppgs']: {cur_video_sample['gt_ppgs'].shape}")
        video_samples.append(cur_video_sample)

    hr_window_size = 300
    stride = 128
    # stride = 1
    mae_list = []
    rmse_list = []
    pcc_list = []
    all_hr_est = []
    all_hr_gt = []
    for index, cur_video_sample in enumerate(video_samples):
        cur_est_ppgs = cur_video_sample['est_ppgs']

        # Load GT
        cur_gt_ppgs = cur_video_sample['gt_ppgs']
        cur_est_ppgs = (cur_est_ppgs - np.mean(cur_est_ppgs)) / np.std(cur_est_ppgs)
        cur_gt_ppgs = (cur_gt_ppgs - np.mean(cur_gt_ppgs)) / np.std(cur_gt_ppgs)

        # Get est HR for each window
        hr_est_temp = []
        hr_gt_temp = []
        for start in range(0, len(cur_est_ppgs) - hr_window_size, stride):
            ppg_est_window = cur_est_ppgs[start:start + hr_window_size]

            ppg_gt_window = cur_gt_ppgs[start:start + hr_window_size]

            # Normalize PPG
            ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
            ppg_gt_window = (ppg_gt_window - np.mean(ppg_gt_window)) / np.std(ppg_gt_window)
            hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            hr_gt_temp.append(pulse_rate_from_power_spectral_density(
                ppg_gt_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

        hr_est_windowed = np.array([hr_est_temp])
        hr_gt_windowed = np.array(hr_gt_temp)
        # print("hr_est_windowed.shape: ", hr_est_windowed.shape, "\thr_gt_windowed.shape: ", hr_gt_windowed.shape)
        # print(hr_est_temp == hr_gt_temp)
        all_hr_est.append(hr_est_temp)
        all_hr_gt.append(hr_gt_temp)

        # Errors
        RMSE, MAE, MAX, PCC = getErrors(hr_est_windowed, hr_gt_windowed)
        # MAE, RMSE, PCC = cal_mae_rmse_r(np.array(hr_est_temp), np.array(hr_gt_temp))

        mae_list.append(MAE)
        rmse_list.append(RMSE)
        pcc_list.append(PCC)
    # print('Mean MAE:', np.mean(np.array(mae_list)))
    return np.array(mae_list), np.array(rmse_list), np.array(pcc_list), (all_hr_est, all_hr_gt)


def get_rjtf(data_f, frame_length, samples):
    sig = torch.from_numpy(data_f).sum(-1)
    win = torch.hamming_window(36)
    # TODO: log magnitude of both \alpha and \theta ? or complex input
    sig_mag = torch.abs(sig) * 1 + 0j
    rjtf_mag = torch.stft(sig_mag, 256, 8, win_length=36, window=win, return_complex=False).permute(2, 1, 0)
    rjtf_mag = (rjtf_mag - rjtf_mag.mean()) / rjtf_mag.std()  # N, T, F
    sig_phase = torch.exp(1j * torch.angle(sig))
    rjtf_phase = torch.stft(sig_phase, 256, 8, win_length=36, window=win, return_complex=False).permute(2, 1, 0)
    rjtf_phase = (rjtf_phase - rjtf_phase.mean()) / rjtf_phase.std()  # N, T, F
    rjtf = torch.cat([rjtf_mag, rjtf_phase])  # 2N , T, F
    rjtf = F.interpolate(rjtf.unsqueeze(0), size=(frame_length, samples), mode="bicubic")[0]
    return rjtf  # 2N，T，F--->4，128，256


# Test Functions
def get_mapped_fitz_labels(fitz_labels_path, session_names):
    with open(fitz_labels_path, "rb") as fpf:
        out = pickle.load(fpf)

    #mae_list
    #session_names
    sess_w_fitz = []
    fitz_dict = dict(out)
    l_m_d_arr = []
    for i, sess in enumerate(session_names):
        pid = sess.split("_")
        pid = pid[0] + "_" + pid[1]
        fitz_id = fitz_dict[pid]
        if(fitz_id < 3):
            l_m_d_arr.append(1)
        elif(fitz_id < 5):
            l_m_d_arr.append(-1)
        else:
            l_m_d_arr.append(2)
    return l_m_d_arr

def eval_clinical_performance(hr_est, hr_gt, fitz_labels_path, session_names):
    l_m_d_arr = get_mapped_fitz_labels(fitz_labels_path , session_names)
    l_m_d_arr = np.array(l_m_d_arr)
    #absolute percentage error
    # print(hr_gt.shape, hr_est.shape)
    apes = np.abs(hr_gt - hr_est)/hr_gt*100
    # print(apes)
    l_apes = np.reshape(apes[np.where(l_m_d_arr==1)], (-1))
    d_apes = np.reshape(apes[np.where(l_m_d_arr==2)], (-1))

    l_5 = len(l_apes[l_apes <= 5])/len(l_apes)*100 
    d_5 = len(d_apes[d_apes <= 5])/len(d_apes)*100
    
    l_10 = len(l_apes[l_apes <= 10])/len(l_apes)*100
    d_10 = len(d_apes[d_apes <= 10])/len(d_apes)*100

    print("AAMI Standard - L,D")
    print(l_10, d_10)

def eval_performance(hr_est, hr_gt):
    hr_est = np.reshape(hr_est, (-1))
    hr_gt  = np.reshape(hr_gt, (-1))
    r = scipy.stats.pearsonr(hr_est, hr_gt)
    mae = np.sum(np.abs(hr_est - hr_gt))/len(hr_est)
    hr_std = np.std(hr_est - hr_gt)
    hr_rmse = np.sqrt(np.sum(np.square(hr_est-hr_gt))/len(hr_est))
    hr_mape = sklearn.metrics.mean_absolute_percentage_error(hr_est, hr_gt)

    return mae, hr_mape, hr_rmse, hr_std, r[0]

def eval_performance_bias(hr_est, hr_gt, fitz_labels_path, session_names):
    l_m_d_arr = get_mapped_fitz_labels(fitz_labels_path , session_names)
    l_m_d_arr = np.array(l_m_d_arr)

    general_performance = eval_performance(hr_est, hr_gt)
    l_p = np.array(eval_performance(hr_est[np.where(l_m_d_arr == 1)], hr_gt[np.where(l_m_d_arr == 1)]))
    d_p = np.array(eval_performance(hr_est[np.where(l_m_d_arr == 2)], hr_gt[np.where(l_m_d_arr == 2)]))

    performance_diffs = np.array([l_p-d_p])
    performance_diffs = np.abs(performance_diffs)
    performance_max_diffs = performance_diffs.max(axis=0)

    print("General Performance")
    print(general_performance)
    print("Performance Max Differences")
    print(performance_max_diffs)
    print("Performance By Skin Tone")
    print("Light - ", l_p)
    print("Dark - ", d_p)

    return general_performance, performance_max_diffs

def get_discriminator_accuracy(y_prob, y_true):
    '''
    Accuracy function for Discriminator
    '''
    assert y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

def eval_fusion_model(dataset_test, model, device = torch.device('cpu'), method = 'both'):
    model.eval()
    print(f"Method : {method}")
    mae_list = []
    session_names = []
    hr_est_arr = []
    hr_gt_arr = []
    hr_rgb_arr = []
    hr_rf_arr = []
    est_wv_arr = []
    gt_wv_arr = []
    rgb_wv_arr = []
    rf_wv_arr = []
    for i in range(len(dataset_test)):
        pred_ffts = []
        targ_ffts = []
        pred_rgbs = []
        pred_rfs  = []
        train_sig, gt_sig = dataset_test[i]
        sess_name = dataset_test.all_combs[i][0]["video_path"]
        session_names.append(sess_name)

        train_sig['est_ppgs'] = torch.tensor(train_sig['est_ppgs']).type(torch.float32).to(device)
        train_sig['est_ppgs'] = torch.unsqueeze(train_sig['est_ppgs'], 0)
        train_sig['rf_ppg'] = torch.tensor(train_sig['rf_ppg']).type(torch.float32).to(device)
        train_sig['rf_ppg'] = torch.unsqueeze(train_sig['rf_ppg'], 0)

        gt_sig = torch.tensor(gt_sig).type(torch.float32).to(device)

        with torch.no_grad():
            if method.lower()  == 'rf':
                # Only RF, RGB is noise
                fft_ppg = model(torch.rand(torch.unsqueeze(train_sig['est_ppgs'], axis=0).shape).to(device), torch.unsqueeze(train_sig['rf_ppg'], axis=0))
            elif method.lower() == 'rgb':
                # Only RGB, RF is randn
                fft_ppg = model(torch.unsqueeze(train_sig['est_ppgs'], axis=0), torch.rand(torch.unsqueeze(train_sig['rf_ppg'], axis=0).shape).to(device))
            else:
                # Both RGB and RF
                fft_ppg = model(torch.unsqueeze(train_sig['est_ppgs'], axis=0), torch.unsqueeze(train_sig['rf_ppg'], axis=0))
        # Reduce the dims
        fft_ppg = torch.squeeze(fft_ppg, 1)


        temp_fft = fft_ppg[0].detach().cpu().numpy()
        temp_fft = temp_fft-np.min(temp_fft)
        temp_fft = temp_fft/np.max(temp_fft)

        # Calculate iffts of original signals
        rppg_fft = train_sig['rppg_fft']
        rppg_mag = np.abs(rppg_fft)
        rppg_ang = np.angle(rppg_fft)
        # Replace magnitude with new spectrum
        lix = dataset_test.l_freq_idx 
        rix = dataset_test.u_freq_idx + 1
        roi = rppg_mag[lix:rix]
        temp_fft = temp_fft*np.max(roi)
        rppg_mag[lix:rix] = temp_fft
        rppg_mag[-rix+1:-lix+1] = np.flip(temp_fft)
        rppg_fft_est = rppg_mag*np.exp(1j*rppg_ang)

        rppg_est = np.real(np.fft.ifft(rppg_fft_est))
        rppg_est = rppg_est[0:300] # The 300 is the same as desired_ppg_length given in the dataloader
        gt_est = np.real(np.fft.ifft(train_sig['gt_fft']))[0:300] #The 300 is the same as desired_ppg_length given in the dataloader

        # Re-normalize
        rppg_est = (rppg_est - np.mean(rppg_est)) / np.std(rppg_est)
        gt_est = (gt_est - np.mean(gt_est)) / np.std(gt_est)

        pred_ffts.append(pulse_rate_from_power_spectral_density(rppg_est, 30, 45, 150))
        targ_ffts.append(pulse_rate_from_power_spectral_density(gt_est, 30, 45, 150))
        pred_rgbs.append(pulse_rate_from_power_spectral_density(train_sig['rgb_true'], 30, 45, 150))
        pred_rfs.append(pulse_rate_from_power_spectral_density(train_sig['rf_true'], 30, 45, 150))

        pred_ffts = np.array(pred_ffts)[:,np.newaxis]
        targ_ffts = np.array(targ_ffts)[:,np.newaxis]
        pred_rgbs = np.array(pred_rgbs)[:,np.newaxis]
        pred_rfs = np.array(pred_rfs)[:,np.newaxis]

        hr_est_arr.append(pred_ffts)
        hr_gt_arr.append(targ_ffts)
        hr_rgb_arr.append(pred_rgbs)
        hr_rf_arr.append(pred_rfs)

        _, MAE, _, _ = getErrors(pred_ffts, targ_ffts, PCC=False)

        mae_list.append(MAE)
        est_wv_arr.append(rppg_est)
        gt_wv_arr.append(gt_est)
        rgb_wv_arr.append(train_sig['rgb_true'])
        rf_wv_arr.append(train_sig['rf_true'])
    return np.array(mae_list), session_names, (hr_est_arr, hr_gt_arr), (est_wv_arr,gt_wv_arr, rgb_wv_arr, rf_wv_arr)

def eval_rgb_model_2(root_dir, session_names, model, sequence_length = 64, 
                   file_name = "rgbd_rgb", ppg_file_name = "rgbd_ppg.npy", device=torch.device('cpu')):
    model.eval()
    video_samples = []
    for cur_session in session_names:
        video_sample = {"video_path" : os.path.join(root_dir, cur_session)}
        video_samples.append(video_sample)

    for cur_video_sample in tqdm(video_samples):
        cur_video_path = cur_video_sample["video_path"]
        cur_est_ppgs = None

        frames = extract_video(path=cur_video_path, file_str=file_name)
        target = np.load(os.path.join(cur_video_path, ppg_file_name))

        for cur_frame_num in range(frames.shape[0]):
            # Preprocess
            cur_frame = frames[cur_frame_num, :, :, :]
            cur_frame_cropped = torch.from_numpy(cur_frame.astype(np.uint8)).permute(2, 0, 1).float()
            cur_frame_cropped = cur_frame_cropped / 255
            # Add the T dim
            cur_frame_cropped = cur_frame_cropped.unsqueeze(0).to(device) 

            # Concat
            if cur_frame_num % sequence_length == 0:
                cur_cat_frames = cur_frame_cropped
            else:
                cur_cat_frames = torch.cat((cur_cat_frames, cur_frame_cropped), 0)

            # Test the performance
            if cur_cat_frames.shape[0] == sequence_length:
                
                # DL
                with torch.no_grad():
                    # Add the B dim
                    cur_cat_frames = cur_cat_frames.unsqueeze(0) 
                    cur_cat_frames = torch.transpose(cur_cat_frames, 1, 2)
                    # Get the estimated PPG signal
                    cur_est_ppg = model(cur_cat_frames)
                    cur_est_ppg = cur_est_ppg.squeeze().cpu().numpy()

                # First sequence
                if cur_est_ppgs is None: 
                    cur_est_ppgs = cur_est_ppg
                else:
                    cur_est_ppgs = np.concatenate((cur_est_ppgs, cur_est_ppg), -1)
    
        # Save
        cur_video_sample['est_ppgs'] = cur_est_ppgs
        cur_video_sample['gt_ppgs'] = target[25:]
    print('All finished!')

    # Estimate using waveforms

    hr_window_size = 300
    stride = 128
    mae_list = []
    all_hr_est = []
    all_hr_gt = []
    for index, cur_video_sample in enumerate(video_samples):
        cur_video_path = cur_video_sample['video_path']
        cur_est_ppgs = cur_video_sample['est_ppgs']

        # Load GT
        cur_gt_ppgs = cur_video_sample['gt_ppgs']
        cur_est_ppgs = (cur_est_ppgs - np.mean(cur_est_ppgs)) / np.std(cur_est_ppgs)
        cur_gt_ppgs = (cur_gt_ppgs - np.mean(cur_gt_ppgs)) / np.std(cur_gt_ppgs)

        # Get est HR for each window
        hr_est_temp = []
        hr_gt_temp = []
        for start in range(0, len(cur_est_ppgs) - hr_window_size, stride):
            ppg_est_window = cur_est_ppgs[start:start + hr_window_size]

            ppg_gt_window = cur_gt_ppgs[start:start + hr_window_size]

            # Normalize PPG
            ppg_est_window = (ppg_est_window - np.mean(ppg_est_window)) / np.std(ppg_est_window)
            ppg_gt_window = (ppg_gt_window - np.mean(ppg_gt_window)) / np.std(ppg_gt_window)
            hr_est_temp.append(pulse_rate_from_power_spectral_density(
                ppg_est_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))
            hr_gt_temp.append(pulse_rate_from_power_spectral_density(
                ppg_gt_window, 30, 45, 150, BUTTER_ORDER=6, DETREND=False))

        hr_est_windowed = np.array([hr_est_temp])
        hr_gt_windowed = np.array(hr_gt_temp)
        all_hr_est.append(hr_est_temp)
        all_hr_gt.append(hr_gt_temp)

        # Errors
        _, MAE, _, _ = getErrors(hr_est_windowed, hr_gt_windowed)

        mae_list.append(MAE)
    print('Mean MAE:', np.mean(np.array(mae_list)))
    return np.array(mae_list), (all_hr_est, all_hr_gt)


if __name__ == "__main__":
    model = FusionModel(frame_depth=4, img_size=128, freq_size=256, time_length=256)
    video_list = ['v_21_1']
    rf_file_list = ['21_1']
    maes_val, (_, _) = eval_model(video_list=video_list, rf_file_list=rf_file_list, model=model)
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from rf.proc import rotateIQ
from dataset import RGBData
from losses.NegPearsonLoss import Neg_Pearson
from losses.SNRLoss import SNRLoss_dB_Signals
from utils.eval import eval_rgb_model
from model import RGBModel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Argparser.
def parseArgs():
    parser = argparse.ArgumentParser(description='Configs for thr fusion train script')

    parser.add_argument('-rgb_dir', '--rgb-data-dir', default="/share2/data/zhouwenqing/UCLA-rPPG/rgb_files", type=str,
                        help="Parent directory containing the folders with rgb data")

    parser.add_argument('-fp', '--fitzpatrick-path', type=str,
                        default="/share2/data/zhouwenqing/UCLA-rPPG/fitzpatrick_labels.pkl",
                        help='Pickle file containing the fitzpatrick labels.')

    parser.add_argument('--folds-path', type=str,
                        default="/share2/data/zhouwenqing/UCLA-rPPG/demo_fold.pkl",
                        help='Pickle file containing the folds.')

    parser.add_argument('--fold', type=int, default=0,
                        help='Fold Number')

    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Device on which the model needs to run (input to torch.device). \
                              Don't specify for automatic selection. Will be modified inplace.")

    parser.add_argument('-ckpts', '--checkpoints-path', type=str,
                        default="./ckpt/rgb",
                        help='Checkpoint Folder.')

    parser.add_argument('--verbose', action='store_true', help="Verbosity.")

    parser.add_argument('--viz', action='store_true', help="Visualize.")

    # Train args
    parser.add_argument('--batch-size', type=int, default=4,
                        help="Batch Size for the dataloaders.")

    parser.add_argument('--num-workers', type=int, default=2,
                        help="Number of Workers for the dataloaders.")

    parser.add_argument('--train-shuffle', action='store_true', help="Shuffle the train loader.")
    parser.add_argument('--val-shuffle', action='store_true', help="Shuffle the val loader.")
    parser.add_argument('--test-shuffle', action='store_true', help="Shuffle the test loader.")

    parser.add_argument('--train-drop', action='store_true', help="Drop the final sample of the train loader.")
    parser.add_argument('--val-drop', action='store_true', help="Drop the final sample of the val loader.")
    parser.add_argument('--test-drop', action='store_true', help="Drop the final sample of the test loader.")

    parser.add_argument('-lr', '--learning-rate', type=float, default=9e-3,
                        help="Learning Rate for the optimizer.")

    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-2,
                        help="Weight Decay for the optimizer.")

    parser.add_argument('--epochs', type=int, default=100, help="Number of Epochs.")

    parser.add_argument('--checkpoint-period', type=int, default=5,
                        help="Checkpoint save period.")

    parser.add_argument('--epoch-start', type=int, default=1,
                        help="Starting epoch number.")

    return parser.parse_args()


def train_model(args, model, datasets):
    # Instantiate the dataloaders
    train_dataloader = DataLoader(datasets["train"], batch_size=args.batch_size,
                                  shuffle=args.train_shuffle, drop_last=args.train_drop,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(datasets["val"], batch_size=args.batch_size,
                                shuffle=args.val_shuffle, drop_last=args.val_drop,
                                num_workers=args.num_workers)
    test_dataloader = DataLoader(datasets["test"], batch_size=args.batch_size,
                                 shuffle=args.test_shuffle, drop_last=args.test_drop,
                                 num_workers=args.num_workers)

    if args.verbose:
        print(f"Number of train iterations : {len(train_dataloader)}")
        print(f"Number of val iterations : {len(val_dataloader)}")
        print(f"Number of test iterations : {len(test_dataloader)}")

    ckpt_path = args.checkpoints_path
    latest_ckpt_path = os.path.join(os.getcwd(), f"{ckpt_path}/latest_context.pth")

    # Train Essentials
    loss_fn1 = nn.MSELoss()
    # loss_fn2 = Neg_Pearson()
    # loss_fn3 = SNRLoss_dB_Signals()
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, epochs=args.epochs, steps_per_epoch=len(train_dataloader))
    # A high number to remember the best Loss.
    best_loss = 1e7

    # Train configurations
    epochs = args.epochs
    checkpoint_period = args.checkpoint_period
    epoch_start = args.epoch_start

    if os.path.exists(latest_ckpt_path):
        print('Context checkpoint exists. Loading state dictionaries.')
        checkpoint = torch.load(latest_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        epoch_start += 1

    if args.verbose:
        print(f"Checkpoint Period={checkpoint_period}. Epoch start = {epoch_start}")

    mae_best_loss = np.inf
    lrs = []
    for epoch in range(epoch_start, epochs + 1):
        # Training Phase
        loss_train = 0
        r_loss = 0
        snr_loss = 0
        no_batches = 0
        print("Starting Epoch: {}".format(epoch))
        print(f"Epoch: {epoch} ; LR: {scheduler.get_last_lr()}")
        for batch, (rgb_sig, ppg_sig) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            model.train()
            
            rgb_sig = rgb_sig.type(torch.float32)/(255)
            rgb_sig = rgb_sig.to(args.device)
            ppg_sig = ppg_sig.type(torch.float32).to(args.device)

            # Predict the PPG signal and find ther loss
            pred_signal = model(rgb_sig)

            # loss
            loss1 = loss_fn1(pred_signal, ppg_sig)
            # loss2 = loss_fn2(pred_signal, ppg_sig)
            # loss3 = loss_fn3(pred_signal, ppg_sig)

            loss = loss1
            # loss = 0.01 * loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            
            lrs.append(scheduler.get_last_lr())
            optimizer.step()
            scheduler.step()

            # Accumulate the total loss
            loss_train += loss.item()
            # r_loss += loss2.item()
            # snr_loss += loss3.item()
            no_batches += 1

        torch.save(model.state_dict(), os.path.join(os.getcwd(), f"{ckpt_path}/{epoch}.pth"))
        # See if best checkpoint
        maes_val, rmses_val, pccs_val, (_, _) = eval_rgb_model(video_list=datasets["val"].video_list, model=model, device=args.device)
        # maes_train, rmses_train, pccs_train, (_, _) = eval_rgb_model(video_list=datasets["train"].video_list, model=model, device=args.device)
        current_loss = np.mean(maes_val)
        if (current_loss < mae_best_loss):
            mae_best_loss = current_loss
            torch.save(model.state_dict(), os.path.join(os.getcwd(), f"{ckpt_path}/best.pth"))
            print("Best checkpoint saved!")
        print("Saved Checkpoint!")

        print(f"Epoch: {epoch} ; Loss: {loss_train / no_batches:.4f}")
        # print(f"Epoch: {epoch} ; Loss: {loss_train / no_batches:.4f}, R Loss: {r_loss / no_batches:.4f}")
        # print(f"Epoch: {epoch} ; Loss: {loss_train / no_batches:.4f}, R Loss: {r_loss / no_batches:.4f}, SNR Loss: {snr_loss / no_batches:.4f}")
        print(f"Epoch: {epoch} ; MAE_val: {np.mean(maes_val):.4f}")
        # print(f"Epoch: {epoch} ; MAE_val: {np.mean(maes_val):.4f}, MAE_train: {np.mean(maes_train):.4f}")
        # print(f"Epoch: {epoch} ; RMSE_val: {np.mean(rmses_val):.4f}, RMSE_train: {np.mean(rmses_train):.4f}")
        # print(f"Epoch: {epoch} ; PCC_val: {np.mean(pccs_val):.4f}, PCC_train: {np.mean(pccs_train):.4f}")
        # SAVE CONTEXT AFTER EPOCH
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_ckpt_path)


def main(args):
    # Import essential info, i.e. destination folder and fitzpatrick label path
    rgb_folder = args.rgb_data_dir
    fitz_labels_path = args.fitzpatrick_path
    ckpt_path = args.checkpoints_path

    with open(args.folds_path, "rb") as fp:
        files_in_fold = pickle.load(fp)

    rgb_train_files = files_in_fold[args.fold]["train"]
    rgb_val_files = files_in_fold[args.fold]["val"]
    rgb_test_files = files_in_fold[args.fold]["test"]

    # print(rgb_val_files[:10])

    # print(f"len(rgb_train_files): {len(rgb_train_files)} len(rgb_val_files): {len(rgb_val_files)} len(rgb_test_files): {len(rgb_test_files)}")

    # Dataset
    train_dataset = RGBData(datapath=rgb_folder, video_list=rgb_train_files)
    val_dataset = RGBData(datapath=rgb_folder, video_list=rgb_val_files)
    test_dataset = RGBData(datapath=rgb_folder, video_list=rgb_test_files)

    # print(f"len(train_dataset): {len(train_dataset)} len(val_dataset): {len(val_dataset)} len(test_dataset): {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=args.train_shuffle, drop_last=args.train_drop,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=args.val_shuffle, drop_last=args.val_drop,
                                num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=args.test_shuffle, drop_last=args.test_drop,
                                 num_workers=args.num_workers)

    # Select the device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    if args.verbose:
        print('Running on device: {}'.format(args.device))

    # Create the checkpoints folder if it does not exist
    os.makedirs(ckpt_path, exist_ok=True)

    # Check if Checkpoints exist
    all_ckpts = os.listdir(ckpt_path)
    if (len(all_ckpts) > 0):
        all_ckpts.sort()
        print(f"Checkpoints already exists at : {all_ckpts}")
    else:
        print("No checkpoints found, starting from scratch!")

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    """
        frame_depth必须能被B*T整除
        img_size不是超参数, 固定为128
        time_length为超参数, 对应FusionDataset当中的frame_length参数, 两者必须保持一致
    """
    model = RGBModel(frame_depth=4, img_size=128, time_length=128).to(args.device)
    train_model(args, model, datasets)


if __name__ == '__main__':
    args = parseArgs()
    main(args)
import numpy as np
import matplotlib.pyplot as plt
import os, time, pickle, json
from glob import glob
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Union, Any
from statistics import mean
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

MEAN = (0.5, 0.5, 0.5,)
STD = (0.5, 0.5, 0.5,)
RESIZE = 256


def read_path(data_path, split) -> List[str]:
    path = os.path.join(data_path, split)
    dataset = []
    for p in glob(path + "/" + "*.jpg"):
        dataset.append(p)
    return dataset


class Transform():
    def __init__(self, resize=RESIZE, mean=MEAN, std=STD):
        self.data_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img: Image.Image):
        return self.data_transform(img)


def _separate(img) -> Tuple[Image.Image, Image.Image]:
    img = np.array(img, dtype=np.uint8)
    h, w, _ = img.shape
    w = int(w / 2)
    return Image.fromarray(img[:, :w, :]), Image.fromarray(img[:, w:, :])


class Dataset(object):
    def __init__(self, files: List[str]):
        self.files = files
        self.trasformer = Transform()

    def _separate(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.files[idx])
        input, output = _separate(img)
        input_tensor = self.trasformer(input)
        output_tensor = self.trasformer(output)
        return input_tensor, output_tensor

    def __len__(self):
        return len(self.files)


def de_norm(img):
    img_ = img.mul(torch.FloatTensor(STD).view(3, 1, 1))
    img_ = img_.add(torch.FloatTensor(MEAN).view(3, 1, 1)).detach().numpy()
    img_ = np.transpose(img_, (1, 2, 0))
    return img_


def evaluate(val_dl, name, G, device):
    with torch.no_grad():
        fig, axes = plt.subplots(6, 8, figsize=(12, 12))
        ax = axes.ravel()
        #         G = load_model(name)
        for input_img, real_img in tqdm(val_dl):
            input_img = input_img.to(device)
            real_img = real_img.to(device)

            fake_img = G(input_img)
            batch_size = input_img.size()[0]
            batch_size_2 = batch_size * 2

            for i in range(batch_size):
                ax[i].imshow(input_img[i].permute(1, 2, 0))
                ax[i + batch_size].imshow(de_norm(real_img[i]))
                ax[i + batch_size_2].imshow(de_norm(fake_img[i]))
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                ax[i + batch_size].set_xticks([])
                ax[i + batch_size].set_yticks([])
                ax[i + batch_size_2].set_xticks([])
                ax[i + batch_size_2].set_yticks([])
                if i == 0:
                    ax[i].set_ylabel("Input Image", c="g")
                    ax[i + batch_size].set_ylabel("Real Image", c="g")
                    ax[i + batch_size_2].set_ylabel("Generated Image", c="r")
            plt.subplots_adjust(wspace=0, hspace=0)
            break


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='GAN model implementation')
    # -- access to dataset
    parser.add_argument('--root_path', default='/content/drive/MyDrive/Data_pix2pix_complet', help='path to dataset')
    # -- parameters
    parser.add_argument('--MEAN', default=(0.5, 0.5, 0.5,), help='mean')
    parser.add_argument('--STD', default=(0.5, 0.5, 0.5,), help='std')
    parser.add_argument('--RESIZE', type=int, default=256, help='resize')
    parser.add_argument('--LAMBDA', type=int, default=100.0, help='lambda value')
    # -- train
    parser.add_argument('--BATCH_SIZE', type=int, default=8, help='Mini-batch size')
    parser.add_argument('--optimizer_g', type=str, default='ADAM', choices=['adam', 'sgd', 'RMSprop'])
    parser.add_argument('--optimizer_d', type=str, default='ADAM', choices=['adam', 'sgd', 'RMSprop'])
    parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--betas', default=(0.5, 0.999), help='initial betas value')
    parser.add_argument('--EPOCH', default=30, type=int, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    # -- conv / deconv layers
    parser.add_argument('--kernel_size', default=3, help='size of kernel')
    parser.add_argument('--pool_size', default=None, help='size of pool')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--resume', default='/content/drive/MyDrive/saving_D30.pth',
                        help='checkpoint to resume training from (default: None)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = load_args()
    print(args)
    root_path = args.root_path
    # root_path = "/content/drive/MyDrive/Dataset_dents"
    train = read_path(data_path=root_path, split="train")
    val = read_path(data_path=root_path, split="val")
    train_ds = Dataset(train)
    val_ds = Dataset(val)

    BATCH_SIZE = args.BATCH_SIZE
    device = args.device
    torch.manual_seed(0)
    np.random.seed(0)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    evaluate(val_dl, 5, trained_G, device)

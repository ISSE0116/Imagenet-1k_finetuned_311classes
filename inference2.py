################################
#                              #
#          Fine-Tuning         #
#          Inference2          #
################################
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torchvision.datasets.utils import download_url
import os
import sys

#input_file = sys.argv[1]
#input_path = os.path.join('input', input_file)

inp_model_path = sys.argv[1]
PATH = os.path.join('./weight_finetuning_path', inp_model_path) 

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        # これを有効にしないと、計算した勾配が毎回異なり、再現性が担保できない。
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# デバイスを選択する。
device = get_device(use_gpu=True)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 311)
model.load_state_dict(torch.load(PATH, map_location = device))

transform = transforms.Compose(
    [
        transforms.Resize(256),  # (256, 256) で切り抜く。
        transforms.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く
        transforms.ToTensor(),  # テンソルにする。
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 標準化する。
    ]
)

transform2 = transforms.ToTensor()

#img = Image.open(input_path)
#inputs = transform(img)
#inputs = inputs.unsqueeze(0).to(device)

def _get_img_paths(img_dir):
    img_dir = Path(img_dir)
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    img_paths = [str(p) for p in img_dir.iterdir() if p.suffix in img_extensions]
    img_paths.sort()

    return img_paths


class ImageFolder(Dataset):
    def __init__(self, img_dir, transform):
        # 画像ファイルのパス一覧を取得する。
        self.img_paths = _get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        inputs = self.transform(img)

        return {"image": inputs, "path": path}

    def __len__(self):
        return len(self.img_paths)


# Dataset を作成する。
dataset = ImageFolder("./input", transform)
# DataLoader を作成する。
dataloader = DataLoader(dataset, batch_size=8)

'''
model.eval()
outputs = model(inputs)

batch_probs = F.softmax(outputs, dim=1)
batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)
'''
def get_classes():
    if not Path("data/imagenet_class_index.json").exists():
        # ファイルが存在しない場合はダウンロードする。
        download_url("https://git.io/JebAs", "./data", "imagenet_class_index.json")

    # クラス一覧を読み込む。
    with open("data/imagenet_class_index.json") as f:
        data = json.load(f)
        class_names = [x["ja"] for x in data]

    return class_names


# クラス名一覧を取得する。

class_names = get_classes()

'''
for probs, indices in zip(batch_probs, batch_indices):
    for k in range(5):
        print(f"Top-{k + 1} {class_names[indices[k]]} {probs[k]:.2%}")
'''

for batch in dataloader:
    inputs = batch["image"].to('cpu')
    outputs = model(inputs)

    batch_probs = F.softmax(outputs, dim=1)

    batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)

    for probs, indices, path in zip(batch_probs, batch_indices, batch["path"]):
        #display.display(display.Image(path, width=224))
        print()
        print(f"path: {path}")
        for k in range(5):
            print(f"Top-{k + 1} {probs[k]:.2%} {indices[k]}")
            print()

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

inp_model_path = sys.argv[1]
inp_modellayer = int(sys.argv[2])
inp_img = sys.argv[3]
weight_dir = '../weight_finetuning_path/weight_finetuning_path_resnet' + str(inp_modellayer)
PATH = os.path.join(weight_dir, inp_model_path) 
img_path = os.path.join('../input', inp_img)

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device(use_gpu=True)

if(inp_modellayer == 18):
    model = models.resnet18(pretrained=True)
if(inp_modellayer == 50):
    model = models.resnet50(pretrained=True)
if(inp_modellayer == 101):
    model = models.resnet101(pretrained=True)
    
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 311)
model.load_state_dict(torch.load(PATH, map_location = device))

transform = transforms.Compose(
    [
        transforms.Resize(256),  
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

transform2 = transforms.ToTensor()

img = Image.open(img_path)
img =  


'''
model.eval()
outputs = model(inputs)

batch_probs = F.softmax(outputs, dim=1)
batch_probs, batch_indices = batch_probs.sort(dim=1, descending=True)
'''

def get_classes():
    if not Path("data/imagenet_class_index.json").exists():
        download_url("https://git.io/JebAs", "../data", "imagenet_class_index.json")

    with open("../data/myclass.json") as f:
        data = json.load(f)
        class_names = [x["en"] for x in data]

    return class_names

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
            print(f"Top-{k + 1} {probs[k]:.2%} {class_names[indices[k]]}")
            print()

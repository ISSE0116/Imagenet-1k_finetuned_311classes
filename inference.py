################################
#                              #
#          Fine-Tuning         #
#           Inference          #
################################

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
import datetime
import sys
from PIL import Image
import json

path = sys.argv[1]
input_path = sys.argv[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = os.path.join('./weight_finetuing_path', path)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 311)
model_ft = model_ft.to(device)
print("学習済み重みをロードしました")

file_path = os.path.join('./input', input_path)
img = Image.open(file_path)
#img = cv2.imread(file_path)
#height, width, channels = img.shape

plt.imshow(img)
plt.show()

#Load modelpath
model_ft.load_state_dict(torch.load(PATH, map_location = device))
model_ft.eval()

#ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))
#predictor = ILSVRPredictor(ILSVRC_class_index)

print(img.size())
inputs = img.unsqueeze_(0)

out = model_ft(inputs)
result = predictor.predict_max(out)

print(result)

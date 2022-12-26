################################
#                              #
#       transfer-learning      #
#         Training_aug         #
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

plt.ion()

#読み取る画像ディレクトリ指定
data_dir = '/mnt/data1/kikuchi/kikuchisan/valval/train'
#data_dir = '/mnt/data1/kikuchi/kikuchisan/t'

batch_size = int(sys.argv[1]) 
num_epochs = int(sys.argv[2])
lr = float(sys.argv[3])
cuda_num = sys.argv[4]

#####パラメータ設定#####
#batch_size = 256 
#num_epochs = 25  
#lr = 0.005
step_size = int(num_epochs * 0.9)
wd = 0.0001 
########################

#lossとaccの遷移を記録、グラフに使う
loss_t=[]
loss_v=[]
acc_t=[]
acc_v=[]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5)
    ]),
}

image_datasets = datasets.ImageFolder(data_dir,data_transforms['train'])

#指定したディレクトリの画像を学習データ、検証データに分割。
train_size = int(0.8 * len(image_datasets))
val_size = len(image_datasets) - train_size
train, val = torch.utils.data.random_split(image_datasets, [train_size, val_size])
print("\n\n\n")
print("[ABOUT]")
print("\033[32m###################################################################\033[0m")
print(f"\033[31mfull: {len(image_datasets)} -> train: {len(train)}, validation: {len(val)}\033[0m")
print("\033[32m###################################################################\033[0m")

print("\033[31mlr(Initial): {}, batch_size: {}, num_epoch: {}, step_size: {}, weight_decay: {}\033[0m".format(lr, batch_size, num_epochs, step_size, wd))
print("\033[32m###################################################################\033[0m")
print("\n\n\n")

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size, shuffle=True)
data_loader = {'train':train_loader,'val':val_loader}

train_sizes = len(train)
val_sizes = len(val)
dataset_sizes = {'train':train_sizes,'val':val_sizes}

class_names = image_datasets.classes
device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")

#visualize_trainingdataset
best_acc = 0.0
dt_now = 0

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0)

#Get a batch of Graining data
inputs, classes = next(iter(train))

#Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])

##########################training moGels##############################
def train_model(model, criterin, optimizer, scheduler, num_epochs):
    since = time.time()

    i = 1
    
    best_models_wts = copy.deepcopy(model.state_dict())

    global loss_t, loss_v, acc_t, acc_v, best_acc, dt_now

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
               model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loader[phase]:          #dataloaderからdataset呼び出し
                inputs = inputs.to(device)                     #GPUに転送
                labels = labels.to(device) 
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'): #train時のみ勾配の算出をオンにするの意
                    #m = nn.dropout(inputs)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    i += 1

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)   #lossとbatchサイズとの掛け算
                running_corrects += torch.sum(preds == labels.data)
            if(phase == 'train'):
                epoch_loss_t = running_loss / dataset_sizes['train']
                epoch_acc_t = running_corrects.double() / dataset_sizes['train']
                loss_t.append(epoch_loss_t)
                acc_t.append(epoch_acc_t.item())
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss_t, epoch_acc_t))
            else:
                epoch_loss_v = running_loss / dataset_sizes['val']
                epoch_acc_v = running_corrects.double() / dataset_sizes['val']
                loss_v.append(epoch_loss_v)
                acc_v.append(epoch_acc_v.item())
                print('{} Loss: {:.4f} '.format(phase, epoch_loss_v), end = '')

                if epoch_acc_v > best_acc:
                    best_acc = epoch_acc_v
                    best_models_wts = copy.deepcopy(model.state_dict())
                    print('\033[32mAcc: {:.4f}\033[0m'.format(epoch_acc_v))
                else:
                    print('Acc: {:.4f}'.format(epoch_acc_v))

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('Best val Acc: {:4f}'.format(best_acc))
    
    dt_now = datetime.datetime.now()
    dt_now = str(dt_now.month) + str(dt_now.day) + '-' + str(dt_now.hour) + str(dt_now.minute) 

    model_path = 'model_path_' + '{}-{}-{}_'.format(lr, batch_size, num_epochs) + dt_now
    torch.save(best_models_wts, os.path.join('../weight_finetuning_path/weight_finetuning_path_resnet18', model_path))
    print()
    print('!!!!!save_{}!!!!!'.format(model_path))
    return model

############################################################################################

#visualize predicted model

def visualize_model(model, num_images=6): 
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader[val]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images// 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
            model.train(mode=was_training)
'''
#finetuning the convert

model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features                                                     #modelの定義
model_ft.fc = nn.Linear(num_ftrs, 311)                                                 #出力層を311classに変更
model_ft = model_ft.to(device)                                                         #GPUに送信
criterion = nn.CrossEntropyLoss()                                                      #損失関数定義(分類なのでクロスエントロピー)
optimizer_ft = optim.SGD(model_ft.parameters(), lr, momentum=0.9, weight_decay=wd)     #最適化手法(SGD)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size, gamma=0.1)             #スケジューラー

#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
#visualize_model(model_ft)

train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs)
'''

#Convnet as fixed feature extractor
model_conv = torchvision.models.resnet50(pretrained = True)
for param in model_conv.parameters():
    param.requires_grad = False
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 311)
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr, momentum=0.9, weight_decay=wd)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size, gamma=0.1) 

train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs)


#plot the result graph
fig = plt.figure(figsize = (10,5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(range(len(loss_t)), loss_t, label="Loss(Train)")
ax1.plot(range(len(loss_v)), loss_v, label="Loss(Val)")
ax2.plot(range(len(acc_t)), acc_t, label="Acc(Train)")
ax2.plot(range(len(acc_v)), acc_v, label="Acc(Val)")
ax1.legend()
ax2.legend()
plt.title('size[ [train]:{}  [val]:{} ]  [lr]:{}  [batch_size]:{}  [epoch]:{}  [best_acc_val]:{:.3f}'.format(train_sizes, val_sizes, lr, batch_size, num_epochs, best_acc), loc = 'right', y = 1.05, weight = 1000, color = 'green')

ax1.set_ylim(0,7)
ax2.set_ylim(0,1)
ax1.set_yticks([0,1,2,3,4,5,6,7])
ax2.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
ax1.grid(which = "major", axis = "y", color = "black", alpha = 0.8, linestyle = "--", linewidth = 0.5)
ax2.grid(which = "major", axis = "y", color = "black", alpha = 0.8, linestyle = "--", linewidth = 0.5)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Acc")
graph = 'train_result_graph_' + '{}-{}-{}_'.format(lr, batch_size, num_epochs) + dt_now + '_aug'  + '.png' 
plt.savefig(os.path.join("../graph/", graph))

print()
print("!!!!!end_to_plot_graph!!!!!")
print()

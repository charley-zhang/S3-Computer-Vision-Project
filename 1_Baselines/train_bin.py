#!/usr/bin/env python
# coding: utf-8


import os, sys
import math
import random
import cv2
import pickle
import numpy as np
import pandas as pd
from PIL import Image

import torch, torchvision
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models, transforms


## Training Consts
MODELS_PATH = '/afs/crc.nd.edu/user/y/yzhang46/_DLResources/Models'
MODELS = {
    'alexnet': os.path.join(MODELS_PATH, 'alexnet.pth'),
    'vgg16': os.path.join(MODELS_PATH, 'vgg16_bn.pth'),
    'resnet18': os.path.join(MODELS_PATH, 'resnet18.pth'),
    'resnet34': os.path.join(MODELS_PATH, 'resnet34.pth'),
    'resnet50': os.path.join(MODELS_PATH, 'resnet50.pth'),
    'resnet101': os.path.join(MODELS_PATH, 'resnet101.pth'),
    'resnet152': os.path.join(MODELS_PATH, 'resnet152.pth'),
    'resnext50': os.path.join(MODELS_PATH, 'resnext50_32x4d.pth'),
    'resnext101': os.path.join(MODELS_PATH, 'resnext101_32x8d.pth'),
    'densenet121': os.path.join(MODELS_PATH, 'densenet121.pth'),
    'densenet201': os.path.join(MODELS_PATH, 'densenet201.pth'),
    'inceptionv3': os.path.join(MODELS_PATH, 'inception_v3.pth'),
}
EXP_NAME = 'bin_v1'
# MODEL_NAMES = ['densenet121','densenet201','vgg16']
MODEL_NAMES = ['densenet201']

DEVICE = torch.device('cuda') 
NUM_EPOCHS = 40
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = .0005
LRD = .25
PATIENCE = 5

## Data Handling Consts
HAM_DIR = '/afs/crc.nd.edu/user/y/yzhang46/_DLResources/Datasets/[C]HAM10000'
ALL_IMG_FPS = [os.path.join(HAM_DIR,'Train',f) for f in os.listdir(os.path.join(HAM_DIR,'Train'))]
ALL_IMG_IDS = [os.path.splitext(os.path.basename(f))[0] for f in ALL_IMG_FPS]

CROP_SIZE = 350
IMG_SIZE = 224
NORM_MEAN = [0.7630423088417134, 0.5456486014607426, 0.5700468609021178]
NORM_STD = [0.0891409288333237, 0.11792632289606514, 0.1324623088597418]
CLASSES_TO_FULLNAMES = {
    'NV': 'Melanocytic nevi',
    'MEL': 'dermatofibroma',
    'BKL': 'Benign keratosis-like lesions ',
    'BCC': 'Basal cell carcinoma',
    'AKIEC': 'Actinic keratoses',
    'VASC': 'Vascular lesions',
    'DF': 'Dermatofibroma'
}
CLASSES = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']


### Data Methods and Containers
class HAM10k(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        X = None
        try:
            X = Image.open(self.df['path'].iloc[idx])
        except:
            pd.options.display.max_colwidth = 100
            print(f"Attempting to open idx {idx}")
            print(self.df.iloc[idx])
        y = torch.tensor(int(self.df['label'].iloc[idx]))
        if self.transform:
            X = self.transform(X)
        return X, y

### Instantiate Data Constants
# Datasets
df_dict = {'id': [], 'label': [], 'path': []}
with open(os.path.join(HAM_DIR,'Labels.csv'),'r') as f:
    for idx, line in enumerate(f):
        if idx == 0: continue
        line = line.rstrip()
        comps = line.split(',')
        for i in range(1,8):
            if '1' in comps[i]:
                df_dict['label'].append(i-1)
                break
        df_dict['id'].append(comps[0])
        df_dict['path'].append(os.path.join(HAM_DIR,'Train',comps[0] + '.jpg'))
DF = pd.DataFrame(df_dict); 

train_ids = []
with open(os.path.join(HAM_DIR,'TrainSplits','train.txt'),'r') as f:
    for line in f:
        train_ids.append(line.rstrip())
TRAIN_DF = DF.loc[DF['id'].isin(train_ids)]
TRAIN_DF = TRAIN_DF.reset_index(drop=True)
train_transform = transforms.Compose([transforms.CenterCrop(CROP_SIZE),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(NORM_MEAN, NORM_STD)])
TRAIN_SET = HAM10k(TRAIN_DF, train_transform)
TRAIN_LOADER = DataLoader(TRAIN_SET,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)


val_ids = []
with open(os.path.join(HAM_DIR,'TrainSplits','val.txt'),'r') as f:
    for line in f:
        val_ids.append(line.rstrip())
val_transform = transforms.Compose([transforms.CenterCrop(CROP_SIZE), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(NORM_MEAN, NORM_STD)])
VAL_DF = DF.loc[DF['id'].isin(val_ids)]
VAL_SET = HAM10k(VAL_DF, val_transform)
VAL_LOADER = DataLoader(VAL_SET,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS)

# Cleanup
del df_dict, train_ids, val_ids


### Model Functions
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Model specific variables
    model_ft = None
    input_size = 0

    if model_name == "vgg16": #VGG w/BN
        # model_ft = models.vgg11_bn(pretrained=use_pretrained)
        model_ft = models.vgg16_bn()
        model_ft.load_state_dict(torch.load(MODELS[model_name]))
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "densenet121": # Dense-121
        # model_ft = models.densenet121(pretrained=use_pretrained)
        import re
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(MODELS[model_name])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model_ft = models.densenet121()
        model_ft.load_state_dict(state_dict)
        
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "densenet201": # Dense-121
        # model_ft = models.densenet121(pretrained=use_pretrained)
        import re
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(MODELS[model_name])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model_ft = models.densenet201()
        model_ft.load_state_dict(state_dict)
        
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        sys.exit()
    return model_ft, input_size


### Statistics and Evaluation
class StatTracker:
    def __init__(self):
        self.iter_train_loss = []
        self.iter_train_acc = []
        self.full_train_acc = []
        self.full_train_loss = []
        self.full_val_acc = []
        self.full_val_loss = []
    def iter_update(self, tloss, tacc):
        self.iter_train_loss.append(tloss)
        self.iter_train_acc.append(tacc)
    def full_update(self, tacc, tloss, vacc, vloss):
        self.full_train_acc.append(tacc)
        self.full_train_loss.append(tloss)
        self.full_val_acc.append(vacc)
        self.full_val_loss.append(vloss)


def validate(tracker, model, criterion, optimizer):
    model.eval()
    sum_vl, sum_vacc = 0., 0.
    with torch.no_grad():
        print('Computing val stats..')
        for i, data in enumerate(VAL_LOADER):
            images, labels = data
            for idx, l in enumerate(labels):
                labels[idx] = 0 if labels[idx] != 0 else 1
            images = Variable(images).to(DEVICE)
            labels = Variable(labels.to(torch.float32)).to(DEVICE)
            N = images.size(0)
            
            output = model(images).squeeze()
            sig = nn.Sigmoid()
            output = sig(output)
            
            prediction = output.round()
            num_correct = prediction.eq(labels.view_as(prediction)).sum().item()

            sum_vl += criterion(output, labels).item()
            sum_vacc += num_correct/N
    vacc, vloss = sum_vacc/len(VAL_LOADER), sum_vl/len(VAL_LOADER)
    return vloss, vacc


### Training Functions

def train_model(train_loader, model, criterion, optimizer, scheduler,
                epochs=10, tracker=None):
    
    for epoch in range(epochs):
        print(f'\n=========\nTraining Epoch {epoch+1} ({model.name})\n=========\n')
        sum_tl, sum_tacc = 0., 0.
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data
            for idx, l in enumerate(labels):
                labels[idx] = 0 if labels[idx] != 0 else 1
            images = Variable(images).to(DEVICE)
            labels = Variable(labels.to(torch.float32)).to(DEVICE)
            N = images.size(0)
            
            optimizer.zero_grad()
            output = model(images).squeeze()
            sig = nn.Sigmoid()
            output = sig(output)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            prediction = output.round()
            num_correct = prediction.eq(labels.view_as(prediction)).sum().item()

            # print(f'sig-out: {output}')
            # print(f'pred:    {prediction}')
            # print(f'lab:     {labels}')
            # print(f'num-correct: {num_correct}')
            # sys.exit()
            
            # Stats and status
            sum_tl += loss.item()
            sum_tacc += num_correct/N
            if i % 100 == 0:
                tacc = num_correct/N
                print(f'[Epoch {epoch+1}], [Iter {i+1}/{len(train_loader)+1}], '
                      f'[TrnLoss {loss.item():.4}], [TrnAcc {tacc:.4}]')
                tracker.iter_update(loss.item(),tacc)
        tloss, tacc = sum_tl/len(train_loader), sum_tacc/len(train_loader)
        vloss, vacc = validate(tracker, model, criterion, optimizer)
        tracker.full_update(tacc, tloss, vacc, vloss)
        scheduler.step(vacc)
        print(f'-------\n[Epoch {epoch+1}], [Train Acc {tacc:.4}, '
                      f'[Val Acc {vacc:.4}], [Val Loss {vloss:.4}]')

def train_models():
    trackers = []
    
    # Train
    for modelname in MODEL_NAMES:
        model_ft, input_size = initialize_model(modelname,
                                                1, # len(CLASSES), 
                                                feature_extract=False, 
                                                use_pretrained=True)
        model = model_ft.to(DEVICE)
        model.name = modelname
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                         verbose=True,
                                                         factor=LRD, 
                                                         patience=PATIENCE)
        criterion = nn.BCELoss().to(DEVICE) #nn.CrossEntropyLoss().to(DEVICE)
        tracker = StatTracker()
        
        train_model(TRAIN_LOADER,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epochs=NUM_EPOCHS,
                    tracker=tracker)
        trackers.append(tracker)
        torch.save({'epoch': NUM_EPOCHS,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    f'./{modelname}-{EXP_NAME}-ep{NUM_EPOCHS}.pth')
    
    return trackers
    


if __name__ == '__main__':
    trackers = train_models()
    with open(f'trackerlist_{EXP_NAME}.pkl', 'wb') as f:
        pickle.dump(trackers, f)






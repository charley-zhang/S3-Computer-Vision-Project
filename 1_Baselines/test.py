import os, sys
from PIL import Image
import cv2
import pickle
import numpy as np
import pandas as pd
import csv

import torch, torchvision
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {DEVICE} device.\n')

# MODEL_FILES = ['./OutModels/vgg16-ep40-decay.pth',
#                './OutModels/densenet121-ep40-decay.pth',
#                './OutModels/densenet201-ep40-decay.pth'
#               ]
MODEL_FILES = ['./densenet121-v3-ep40-decay-balancedtrainset.pth']

IMG_SIZE = 224
NORM_MEAN = [0.7630423088417134, 0.5456486014607426, 0.5700468609021178]
NORM_STD = [0.0891409288333237, 0.11792632289606514, 0.1324623088597418]

HAM_DIR = '/afs/crc.nd.edu/user/y/yzhang46/_DLResources/Datasets/[C]HAM10000'
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
TEST_IMAGES = [os.path.join(HAM_DIR,'Test',f) for f in os.listdir(os.path.join(HAM_DIR,'Test'))]
TEST_IMG_IDS = [os.path.splitext(os.path.basename(f))[0] for f in TEST_IMAGES]


class HAM10k(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        X = Image.open(self.df['path'].iloc[idx])
        y = self.df['id'].iloc[idx]
        if self.transform:
            X = self.transform(X)
        return X, y

df_dict = {'id': [], 'path': []}
for c in CLASSES:
    df_dict[c] = [0]*len(TEST_IMAGES)
for i in range(len(TEST_IMAGES)):
    df_dict['id'].append(TEST_IMG_IDS[i])
    df_dict['path'].append(TEST_IMAGES[i])
TEST_DF = pd.DataFrame(df_dict)
test_transform = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(NORM_MEAN, NORM_STD)])
TEST_SET = HAM10k(TEST_DF, test_transform)
TEST_LOADER = DataLoader(TEST_SET,
                        batch_size=32,
                        shuffle=False,
                        num_workers=0)

def initialize_test_model(model_path):
    model_ft = None
    if 'vgg' in model_path:
        model_ft = models.vgg16_bn()
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,7)
        model_ft.load_state_dict(torch.load(model_path)['model_state_dict'])
    elif 'densenet121' in model_path: # Dense-121
        import re
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(model_path)['model_state_dict']
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model_ft = models.densenet121()
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs,7)
        model_ft.load_state_dict(state_dict)
    elif 'densenet201' in model_path: # Dense-201
        import re
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(model_path)['model_state_dict']
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model_ft = models.densenet201()
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs,7)
        model_ft.load_state_dict(state_dict)
    else:
        print("Invalid model name, exiting...")
        sys.exit()
    return model_ft


if __name__ == '__main__':

    for model_file in MODEL_FILES:
        modelname = os.path.basename(model_file).split('.')[0]
        print(f'----\nTesting Model: ({modelname})')
        with open(f'pred_{modelname}.csv','w+') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', 
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['image'] + CLASSES)
            model = initialize_test_model(model_file).to(DEVICE)
            with torch.no_grad():
                for i, data in enumerate(TEST_LOADER):
                    images, ids = data
                    images = Variable(images).to(DEVICE)
                    outputs = model(images)
                    prediction = outputs.max(1, keepdim=True)[1].squeeze()
                    print(f'--\nIter {i} (Data Shape: {images.shape}), \n'
                          f'Preds: {prediction} \n')
                    for bnum, imgID in enumerate(ids):
                        labs = [0.]*len(CLASSES)
                        labs[prediction[bnum]] = 1.0
                        writer.writerow([imgID] + labs)





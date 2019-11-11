import os, sys
import cv2
import pickle
import numpy as np
import pandas as pd
import csv
from PIL import Image
from tqdm import tqdm

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
BATCH_SIZE = 32

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
        y = (torch.tensor(int(self.df['label'].iloc[idx])), 
             self.df['id'].iloc[idx])
        if self.transform:
            X = self.transform(X)
        return X, y

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



############### Graphing Stats ###################
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

def graph():
    with open('tracker_list.pkl','rb') as f:
        tl = pickle.load(f)

class Eval:
    def __init__(self, modelname, modelpath, classes):
        self.modelname = modelname
        self.modelpath = modelpath
        self.classes = classes
        self.ids, self.predictions, self.labels = [], None, None
    def update(self, ids, predictions, labels):
        predictions, labels = predictions.cpu().numpy(), labels.cpu().numpy()
        self.ids += ids
        self.predictions = predictions if (self.predictions is None) else np.concatenate((self.predictions, predictions))
        self.labels = labels if (self.labels is None) else np.concatenate((self.labels, labels))
    def printOverview(self):
        print(f"Overall Acc = {self.getAccuracy()} ({len(self.getPredictionsLabels(res='right')[0])}/{len(self.getPredictionsLabels()[0])})")
        balAcc = 0.
        for c in self.classes:
            numRight = len(self.getPredictionsLabels(cls=c,res='right')[0])
            numTot = len(self.getPredictionsLabels(cls=c)[0])
            balAcc += numRight/numTot
            print(f"Class {c}, Acc = {self.getAccuracy(cls=c):.3}"
                  f" ({numRight}/{numTot})")
        print(f"*Balanced Accuracy = {balAcc/len(self.classes)}")
    def getPredictionsLabels(self, cls=None, res='all'): # res = 'all', 'right', 'wrong'
        idxs = None
        if cls is None:
            if res == 'all':
                return self.predictions, self.labels
            else:
                idxs = np.where(self.predictions == self.labels) if res=='right' else \
                       np.where(self.predictions != self.labels)
        elif cls in self.classes:
            idxs = np.where(self.labels == self.classes.index(cls))
            if res == 'right':
                idxs = np.where(self.predictions[idxs] == self.labels[idxs])
            elif res != 'all':
                idxs = np.where(self.predictions[idxs] != self.labels[idxs])
        ret = None if idxs is None else (self.predictions[idxs], self.labels[idxs])
        return ret
    def getAccuracy(self, cls=None):
        if cls is None:
            diff = self.predictions - self.labels
            return np.count_nonzero(diff==0)/len(self.ids)
        elif cls in self.classes:
            idxs = np.where(self.labels == self.classes.index(cls))
            diff = self.predictions[idxs] - self.labels[idxs]
            return np.count_nonzero(diff==0)/diff.shape[0]



def validate(models):
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

    val_ids = []
    with open(os.path.join(HAM_DIR,'TrainSplits','val.txt'),'r') as f:
        for line in f:
            val_ids.append(line.rstrip())
    val_transform = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(NORM_MEAN, NORM_STD)])
    VAL_DF = DF.loc[DF['id'].isin(val_ids)]
    VAL_SET = HAM10k(VAL_DF, val_transform)
    VAL_LOADER = DataLoader(VAL_SET,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=0)
    evals = []
    for model_file in models:
        modelname = os.path.basename(model_file).split('.')[0]
        print(f'\n==========\nValidating Model: ({modelname})\n==========')
        model = initialize_test_model(model_file).to(DEVICE)
        evalObj = Eval(modelname, model_file, CLASSES)
        with torch.no_grad():
            for i, data in enumerate(tqdm(VAL_LOADER)):
                images, y = data
                labels = y[0]
                ids = y[1]
                images = Variable(images).to(DEVICE)
                #labels = Variable(labels).to(DEVICE)
                outputs = model(images)
                prediction = outputs.max(1, keepdim=True)[1].squeeze()
                # print(f'--\nIter {i} (Data Shape: {images.shape}), \n'
                #       f'Preds: {prediction} \n'
                #       f'Label: {labels} \n')
                evalObj.update(ids, prediction, labels)
        evals.append(evalObj)
        evalObj.printOverview()
    return evals

if __name__ == '__main__':
    # evals = validate(['./OutModels/densenet201-ep40-decay.pth'])
    evals = validate(MODEL_FILES)

    








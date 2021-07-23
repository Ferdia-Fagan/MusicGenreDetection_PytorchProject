import os

import numpy as np

import torch
from torch.utils.data import DataLoader,Dataset

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

# I followed https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/5 advice

class MusicDataset(Dataset):
    def __init__(self, rawMelSpectogramsData_X, labelsData_y):
        self.rawMelSpectogramsData_X = rawMelSpectogramsData_X
        self.labelsData_y = labelsData_y

    def __getitem__(self, i):
        return self.rawMelSpectogramsData_X[i],self.labelsData_y[i]

    def __len__(self):
        return self.rawMelSpectogramsData_X.shape[0]


def get_EntireDataSet(rootDir):
    rawMelSpectograms_x = []
    labels_y = []
    for genre_label in next(os.walk(rootDir))[1]:
        for rawMelSpectogramLocation in os.listdir(os.path.join(rootDir, f"{genre_label}")):
            try:
                rawMelSpectogram = np.load(f"{rootDir}/{genre_label}/{rawMelSpectogramLocation}")
                rawMelSpectograms_x.append(rawMelSpectogram)
                labels_y.append(genre_label)
            except:
                print("failed at: ",rawMelSpectogramLocation )

    labelEncoder = LabelEncoder()
    labels_y = labelEncoder.fit_transform(labels_y)

    rawMelSpectograms_x = np.stack(rawMelSpectograms_x)
    labels_y = np.stack(labels_y).squeeze().reshape(-1,1)   # I had to do this so I could stack x and y... Definatly a better way

    return rawMelSpectograms_x,labels_y


def get_TrainAndTestDatasets(dataSetRootFolder, batchSize, testSize, validationSize):
    rawMelSpectograms_x,labels_y = get_EntireDataSet(dataSetRootFolder)

    train_idx, test_idx = train_test_split(list(range(len(rawMelSpectograms_x))), test_size=testSize, shuffle=True)

    train_idx, val_idx = train_test_split(train_idx,test_size=validationSize, shuffle=True)

    rawMelSpectograms_x_train_data,labels_y_train_data = [rawMelSpectograms_x[i] for i in train_idx],[labels_y[i] for i in train_idx]
    rawMelSpectograms_x_val_data,labels_y_val_data = [rawMelSpectograms_x[i] for i in val_idx],[labels_y[i] for i in val_idx]
    rawMelSpectograms_x_test_data,labels_y_test_data = [rawMelSpectograms_x[i] for i in test_idx],[labels_y[i] for i in test_idx]


    # Normalize all data based off training data
    mean = np.mean(rawMelSpectograms_x_train_data)
    std = np.std(rawMelSpectograms_x_train_data)

    rawMelSpectograms_x_train_data = (rawMelSpectograms_x_train_data-mean)/std
    rawMelSpectograms_x_val_data = (rawMelSpectograms_x_val_data-mean)/std
    rawMelSpectograms_x_test_data = (rawMelSpectograms_x_test_data-mean)/std

    train_data = MusicDataset(rawMelSpectograms_x_train_data, labels_y_train_data)
    val_data = MusicDataset(rawMelSpectograms_x_val_data, labels_y_val_data)
    test_data = MusicDataset(rawMelSpectograms_x_test_data, labels_y_test_data)

    train_data_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_data, batch_size=batchSize, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_data, batch_size=batchSize, shuffle=True, drop_last=True)

    return train_data_loader,val_data_loader,test_data_loader

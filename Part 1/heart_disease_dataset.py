import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



class HeartDiseaseDataset(Dataset):
    def __init__(self, path):
        dataframe = pd.read_csv(path)
        self.df = dataframe.drop(columns=['HeartDisease'])
        self.targets = dataframe['HeartDisease']

        scaler = MinMaxScaler()
        scaler.fit(self.df)
        self.df = scaler.transform(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = torch.tensor(self.df[idx]).to(torch.float32)
        target = torch.tensor(self.targets[idx]).to(torch.float32)

        return sample, target


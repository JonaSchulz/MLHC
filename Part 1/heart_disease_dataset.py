import torch
from torch.utils.data import Dataset
import pandas as pd


class HeartDiseaseDataset(Dataset):
    def __init__(self, path):
        dataframe = pd.read_csv(path)
        self.df = dataframe.drop(columns=['HeartDisease'])
        self.targets = dataframe['HeartDisease']
        print("hello")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = torch.tensor(self.df.iloc[idx].values).to(torch.float32)
        target = torch.tensor(self.targets.iloc[idx]).to(torch.float32)

        return sample, target


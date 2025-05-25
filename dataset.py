from torch.utils.data import Dataset
import torch
import os
import pandas as pd
from PIL import Image  # Instead of skimage

class PokemonsData(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        label_names = self.annotations.iloc[:, 1].unique()
        self.label_to_index = {name: idx for idx, name in enumerate(label_names)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Use PIL to ensure RGB format
        image = Image.open(img_path).convert("RGB")

        label_str = self.annotations.iloc[idx, 1]
        label = torch.tensor(int(label_str), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


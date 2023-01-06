"""Dataset Classes"""

import os
import re
import torch
from PIL import Image

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, root, pattern = 'r"(.*)_(.*).png"'):
        self.root = root
        self.pattern = re.compile(pattern)
        self.files = []
        for file in os.listdir(root):
            self.files.append(file)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        file_path = os.path.join(self.root, file)
        image = Image.open(file_path)
        image = image.convert("RGB")
        match = self.pattern.search(file)
        theta = float(match.group(1))
        phi = float(match.group(2))
        return image, theta, phi

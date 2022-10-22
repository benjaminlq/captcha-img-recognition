from logging import raiseExceptions
import numpy as np
import glob
import os
from typing import Union, Optional, Sequence
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
from pathlib import Path
from config import DATA_PATH, NUM_WORKERS, DICTIONARY_PATH
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CaptchaDataset(Dataset):
    def __init__(self, 
                 image_dir: Union[Path, str] = DATA_PATH,
                 resize: Optional[Sequence] = None,
                 ):
        
        self.image_paths = glob.glob(os.path.join(image_dir, "*.png"))          
            
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        if resize:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    transforms.Resize(resize),
                ]
            )
            
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
            
    def setup(self):
        self.sequences = [path.split("/")[-1][:-4] for path in self.image_paths]
        self.vocab_size = 0
        self.char2id = {"*":0}
        self.id2char = {0:"*"}
        
        for sequence in self.sequences:
            for char in sequence:
                if char not in self.char2id:
                    self.vocab_size += 1
                    self.char2id[char] = self.vocab_size
                    self.id2char[self.vocab_size] = char
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        ### Preprocess Image
        image = Image.open(self.image_paths[item]).convert("RGB")
        image = np.array(image)
        image = self.transforms(image)
        
        target = [self.char2id[char] for char in self.sequences[item]]
        
        return image, torch.tensor(target, dtype = torch.long), self.sequences[item]
    
    def sample_image(self):
        sample_idx = np.random.randint(0, len(self.sequences))
        return ## Do something here
    
    def save_dataset(self, path):
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()
        
    def load_dataset(self, path):
        f = open(path, "rb")
        dataset = pickle.load(f)
        f.close()
        return dataset
        
class CaptchaDataloader:
    def __init__(
        self,
        data_dir: Union[Path, str, Dataset] = DATA_PATH,
        batch_size: int = 32,
        val_split: float = 0.2,
        resize: Optional[Sequence] = None,
        num_workers = NUM_WORKERS
        ):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if isinstance(data_dir, Dataset):
            self.full_dataset = data_dir
        else:
            self.full_dataset = CaptchaDataset(data_dir, resize = resize)
        self.full_dataset.setup()
        self.full_dataset.save_dataset(str(DICTIONARY_PATH/"decode_dict.pkl"))
        
        if val_split > 0:
            full_size = len(self.full_dataset)
            test_size = int(np.floor(full_size * val_split))
            train_size = full_size - test_size 
            self.train_dataset, self.test_dataset = random_split(self.full_dataset,
                                                                 [train_size, test_size])
        
        else:
            self.train_dataset = self.full_dataset
            self.test_dataset = None
        
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size =  self.batch_size,
                          shuffle = True, drop_last= True)
    
    def val_loader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size = self.batch_size,
                              shuffle = False, drop_last = False)
        else:
            raiseExceptions("Validation Split not setup")
        
if __name__ == "__main__":
    dataset = CaptchaDataset()
    dataset.setup()
    print(dataset.vocab_size)
    dataloader = CaptchaDataloader()

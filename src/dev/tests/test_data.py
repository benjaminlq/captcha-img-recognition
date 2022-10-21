from config import DATA_PATH, IMG_HEIGHT, IMG_WIDTH, NUM_WORKERS
from dev.dataset import CaptchaDataset, CaptchaDataloader
import torch
import glob
import os

def test_dataset():
    ### Load Dataset
    check_set = CaptchaDataset(image_dir = DATA_PATH,
                              resize = (IMG_HEIGHT, IMG_WIDTH))
    check_set.setup()
    
    image_paths = glob.glob(os.path.join(DATA_PATH, "*.png"))
    
    ### Test length
    assert len(check_set) == len(image_paths), "Dataset size mismatch"
    assert len(check_set.char2id.keys()) == check_set.vocab_size + 1, "Vocab Size Mismatch"
    
    ### Test sample dimension
    image, char_ids, label = check_set[1]
    sequence = "".join([check_set.id2char[idx] for idx in char_ids.tolist()])
    assert sequence == check_set.image_paths[1].split("/")[-1][:-4], "Wrong label assigned"
    assert image.size() == torch.Size([3, IMG_HEIGHT, IMG_WIDTH]), "Wrong Sample Size"
    assert len(label) == 5, "Wrong Captcha Length"
    
def test_dataloader():
    batch_size = 32
    val_split = 0.2
    h, w = 100, 300
    
    check_loader = CaptchaDataloader(data_dir = DATA_PATH,
                                     batch_size = batch_size,
                                     val_split = val_split,
                                     resize = (h, w),
                                     num_workers = NUM_WORKERS,)
    
    ### Test Correct Split Size
    assert len(check_loader.full_dataset) == 1040, "Wrong Full Dataset Size"
    assert len(check_loader.train_dataset) == 832, "Wrong TrainLoader Size"
    assert len(check_loader.test_dataset) == 208, "Wrong TestLoader Size"

    ### Check Minibatch Dimension
    train_loader = check_loader.train_loader()
    test_loader = check_loader.train_loader()  
    train_images, train_targets, train_labels = next(iter(train_loader))
    test_images, test_targets, test_labels = next(iter(test_loader))
    assert train_images.size() == torch.Size([batch_size, 3, h, w]), "Incorrect Train Image Batch Size"
    assert len(train_targets) == batch_size, "Incorrect Train Targets Batch Size"
    assert len(train_labels) == batch_size, "Incorrect Train Labels Batch Size"
    assert len(train_labels[0]) == 5, "Incorrect Output Sequence Length"
    assert test_images.size() == torch.Size([batch_size, 3, h, w]), "Incorrect Test Image Batch Size"
    assert len(test_targets) == batch_size, "Incorrect Test Targets Batch Size"
    assert len(test_labels) == batch_size, "Incorrect Test Labels Batch Size"
    
if __name__ == "__main__":
    test_dataset()
    test_dataloader()
import argparse
from dev.models import CTC
from dev.dataset import CaptchaDataloader
import torch
import torch.nn as nn
import config
from tqdm import tqdm
from config import DEVICE
import matplotlib.pyplot as plt
from dev import utils

def train(model, no_epochs, train_loader, val_loader, learning_rate, decode_dict):
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
    model.to(DEVICE)
    
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(no_epochs):
        epoch_loss = 0
        model.train()
        tk_train = tqdm(train_loader, total=len(train_loader))
        for images, targets, _ in tk_train:
            optimizer.zero_grad()
            log_probs = model(images.to(DEVICE))
            log_probs = log_probs.permute(1,0,2)
            input_lengths = torch.full(
                size = (log_probs.size(1),),
                fill_value = log_probs.size(0),
                dtype = torch.int32,
            )
            
            target_lengths = torch.full(
                size = (targets.size(0),),
                fill_value = targets.size(1),
                dtype = torch.int32,
            )
            
            loss = nn.CTCLoss(blank=0)(log_probs,
                                       targets.to(DEVICE),
                                       input_lengths.to(DEVICE),
                                       target_lengths.to(DEVICE),
                                       )
            
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        val_loss, val_accuracy = eval(model, val_loader, decode_dict)           
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss/len(train_loader)}, Val Loss = {val_loss}, Val Accuracy = {val_accuracy}")
        train_loss_history.append(epoch_loss/len(train_loader))
        val_loss_history.append(val_loss)
    
    plt.figure()
    plt.plot(train_loss_history, color = "red", label = "Training Loss")
    plt.plot(val_loss_history, color = "blue", label = "Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Val Loss vs Epoch")
    plt.legend()
    plt.savefig("History.png")    
    
def eval(model, val_loader, decode_dict):
    model.eval()
    val_loss = 0
    correct_count = 0
    total_count = 0
    tk_eval = tqdm(val_loader, total = len(val_loader))
    with torch.no_grad():
        for images, targets, labels in tk_eval:
            log_probs = model(images.to(DEVICE))
            log_probs = log_probs.permute(1,0,2)
            input_lengths = torch.full(
                size = (log_probs.size(1),),
                fill_value = log_probs.size(0),
                dtype = torch.int32,
            )
            target_lengths = torch.full(
                size = (targets.size(0),),
                fill_value = targets.size(1),
                dtype = torch.int32,
            )
            
            loss = nn.CTCLoss(blank=0)(log_probs,
                                       targets.to(DEVICE),
                                       input_lengths.to(DEVICE),
                                       target_lengths.to(DEVICE)
                                       )
            val_loss += loss.item()
            _, preds = utils.greedy_decode(log_probs, decode_dict)
            for i in range(len(labels)):
                total_count += 1
                if labels[i] == preds[i]:
                    correct_count += 1
            
    val_accuracy = correct_count/total_count * 100
    
    return val_loss, val_accuracy
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-bs", "--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("-lr", "--learning_rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("-ep", "--epochs", type=int, default=config.EPOCHS)
    
    args = parser.parse_args()
    
    data_loader = CaptchaDataloader(batch_size=args.batch_size,
                                    resize = (config.IMG_HEIGHT, config.IMG_WIDTH))
    train_loader = data_loader.train_loader()
    val_loader = data_loader.val_loader()
    model = CTC(data_loader.full_dataset.vocab_size)
    
    train(model, args.epochs, train_loader, val_loader, args.learning_rate,
          data_loader.full_dataset.id2char)
    
import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    loop = tqdm(dataloader, desc="Training from batches:", leave=False)
    for rgb, mask in loop:
        rgb = rgb.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(rgb)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)


def train_one_epoch_mid_fusion(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    loop = tqdm(dataloader, desc="Training from batches:", leave=False)
    for rgb_dsm, mask in loop:
        rgb_dsm = rgb_dsm.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(rgb_dsm)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for rgb, dsm, mask in tqdm(loader):
        rgb, dsm, mask = rgb.to(device), dsm.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(rgb, dsm)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * rgb.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for rgb, dsm, mask in loader:
            rgb, dsm, mask = rgb.to(device), dsm.to(device), mask.to(device)
            logits = model(rgb, dsm)
            loss = criterion(logits, mask)
            running_loss += loss.item() * rgb.size(0)
    return running_loss / len(loader.dataset)

def train_model(model, train_dataset, val_dataset, epochs=50, batch_size=8, lr=1e-4, device='cuda'):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    os.makedirs("experiments/checkpoints", exist_ok=True)

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"experiments/checkpoints/best_model.pth")
            print("Saved best model checkpoint")
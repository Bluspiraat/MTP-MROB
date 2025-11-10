import torch
from tqdm import tqdm
from .metrics import dice_score_multiclass


def validate_one_epoch(model, dataloader, criterion, device, num_classes):

    cce_criterion = nn.CrossEntropyLoss().to(device)

    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_cce = 0.0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)
        for rgb, mask in loop:
            rgb = rgb.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)

            logits = model(rgb)
            loss = criterion(logits, mask)
            running_loss += loss.item()

            cce_loss = cce_criterion(logits, mask)
            running_cce += cce_loss.item()

            dice = dice_score_multiclass(logits, mask, num_classes)
            running_dice += dice

    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    avg_cce =  running_cce / len(dataloader)
    return avg_loss, avg_dice, avg_cce

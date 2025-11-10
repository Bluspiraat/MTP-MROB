from datasets import RGBDataset, RGBDSMDataset
from datasets.augmentations import get_image_net_normalization, get_rgb_transform, get_geometric_transform
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from models import RGBUNet, EarlyFusionUNet
from train import metrics
import torch
from train.loss import ComboLoss
from train.train import train_one_epoch
from train.validate import validate_one_epoch
import json
from tqdm import tqdm

if __name__ == '__main__':

    # Load data
    train_folder = "/home/s2277093/MTP-Data/dataset_diverse_2022_512_sep/train"
    val_folder = "/home/s2277093/MTP-Data/dataset_diverse_2022_512_sep/val"

    num_epochs = 75
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 30
    improvement_threshold = 1e-4
    save_name = "rgbdsm_u_net_r34_b16_a03"
    subset_size = 10000
    batch_size = 16
    alpha = 0.3
    learning_rate = 1e-3    
    scheduler_temp = 50

    # Load dataset
    train_set = RGBDSMDataset(rgb_dir=f'{train_folder}/ortho/',
                            dsm_dir=f'{train_folder}/dsm/',
                            mask_dir=f'{train_folder}/brt/',
                            normalization=get_image_net_normalization(),
                            geo_transform=get_geometric_transform(),
                            rgb_transform=get_rgb_transform())
    val_set = RGBDSMDataset(rgb_dir=f'{val_folder}/ortho/',
                            dsm_dir=f'{val_folder}/dsm/',
                            mask_dir=f'{val_folder}/brt/',
                            normalization=get_image_net_normalization())

    # Dataloaders
    train_batches = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=2, pin_memory=True)
    val_batches = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=2, pin_memory=True)

    # Setup model
    model = EarlyFusionUNet(pretrained=False)

    device = torch.device("cuda:1")  # use second GPU
    model.to(device)

    criterion = ComboLoss(num_classes=14, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_temp, eta_min=1e-5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': [],
        'val_cce': []
    }

    try:
        for epoch in range(num_epochs):
            tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = train_one_epoch(model, train_batches, criterion, optimizer, device)
            val_loss, val_dice, val_cce = validate_one_epoch(model, val_batches, criterion, device, num_classes=14)
            scheduler.step()
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            history['val_cce'].append(val_cce)
            tqdm.write(f"\nEpoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}, Val Dice={val_cce:.4f}")

            if best_val_loss - val_loss > improvement_threshold:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{save_name}.pth")
                tqdm.write(f"\n✅ Model improved (val_loss={val_loss:.4f}), saved checkpoint.")
                with open(f"{save_name}.json", 'w') as f:
                    json.dump(history, f)
            else:
                patience_counter += 1
                with open(f"{save_name}.json", 'w') as f:
                    json.dump(history, f)
                if patience_counter >= patience_limit:
                    tqdm.write("Early stopping triggered.")
                    break

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving progress...")
        torch.save(model.state_dict(), f"{save_name}.pth")
        with open(f"{save_name}.json", 'w') as f:
            json.dump(history, f)


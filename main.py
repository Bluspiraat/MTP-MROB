from datasets import RGBDataset, RGBDSMDataset
from datasets.augmentations import get_image_net_normalization, get_rgb_transform, get_geometric_transform
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from models import RGBUNet
from train import metrics
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from train.loss import ComboLoss
from train.train import train_one_epoch
from train.validate import validate_one_epoch
import json
from tqdm import tqdm
import os


def show_class_image(class_data, title):
    colors = [
        "white", "black", "dimgray", "darkgray", "darkmagenta",
        "darkred", "firebrick", "orange", "forestgreen", "lightgreen",
        "darkorchid", "darkkhaki", "khaki", "lightskyblue", "peru"
    ]
    # labels = colors
    cmap = mcolors.ListedColormap(colors)

    plt.imshow(class_data, cmap, vmin=0, vmax=13)
    plt.title(title)


def create_datasets_splits(folders, splits, seed, DSM=False):
    datasets = []
    if DSM:
        for folder in folders:
            datasets.append(RGBDSMDataset(rgb_dir=f'{folder}/ortho/',
                                 dsm_dir=f'{folder}/dsm/',
                                 mask_dir=f'{folder}/brt/',
                                 normalization=get_image_net_normalization(),
                                 geo_transform=get_geometric_transform(),
                                 rgb_transform=get_rgb_transform()))
    else:
        for folder in folders:
            datasets.append(RGBDataset(rgb_dir=f'{folder}/ortho/',
                                 mask_dir=f'{folder}/brt/',
                                 normalization=get_image_net_normalization(),
                                 geo_transform=get_geometric_transform(),
                                 rgb_transform=get_rgb_transform()))
    train_sets = []
    validation_sets = []
    test_sets = []
    for dataset in datasets:
        train_set_temp, validation_set_temp, test_set_temp = random_split(dataset, splits,
                                                                          generator=torch.Generator().manual_seed(seed))
        train_sets.append(train_set_temp)
        validation_sets.append(validation_set_temp)
        test_sets.append(test_set_temp)
    return ConcatDataset(train_sets), ConcatDataset(validation_sets), ConcatDataset(test_sets)



if __name__ == '__main__':

    # Load data
    data_folders = ["C:/MTP-Data/dataset_diverse_2022_512/bies_bosch",
                    "C:/MTP-Data/dataset_diverse_2022_512/schoorl",
                    "C:/MTP-Data/dataset_diverse_2022_512/vierhouten",
                    "C:/MTP-Data/dataset_diverse_2022_512/soesterberg"]
    rgb_folder = "C:/MTP-Data/dataset_twente_512/ortho/"
    mask_folder = "C:/MTP-Data/dataset_twente_512/brt/"
    class_map = "Data/BRT/class_map.json"

    num_epochs = 20
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    improvement_threshold = 1e-3
    save_name = "rgb_unet_varied_full"
    subset_size = 10000
    batch_size = 4
    alpha = 0.5
    learning_rate = 1e-4
    splits = [0.8, 0.1, 0.1]
    split_seed = 43

    # Split datasets
    train_set, val_set, test_set = create_datasets_splits(data_folders, splits, split_seed)
    train_batches = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True)
    validation_batches = DataLoader(val_set, batch_size=batch_size, shuffle=True , num_workers=4, prefetch_factor=2, pin_memory=True)
    test_batches = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2, pin_memory=True)

    # Setup model
    model = RGBUNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = ComboLoss(num_classes=14, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice': []
    }

    try:
        for epoch in range(num_epochs):
            tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = train_one_epoch(model, train_batches, criterion, optimizer, device)
            val_loss, val_dice = validate_one_epoch(model, test_batches, criterion, device, num_classes=14)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            tqdm.write(f"\nEpoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}")

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


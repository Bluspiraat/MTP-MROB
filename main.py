from datasets import RGBDataset
from torch.utils.data import DataLoader, random_split, Subset
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

if __name__ == '__main__':

    # Load data
    rgb_folder = "C:/MTP-Data/dataset_twente_512/ortho/"
    mask_folder = "C:/MTP-Data/dataset_twente_512/brt/"
    class_map = "Data/BRT/class_map.json"
    dataset = RGBDataset(rgb_folder, mask_folder)

    num_epochs = 20
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    improvement_threshold = 1e-3
    save_name = "rgb_unet_20_10000"
    subset_size = 10000
    alpha = 0.5
    learning_rate = 1e-4

    # Split datasets
    train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(43))

    # Subset train and test set
    subset_indices_train = list(range(4*subset_size))
    train_subset = Subset(train_set, subset_indices_train)
    train_batches = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=4)
    subset_indices_val_test = list(range(4*int(subset_size/10)))
    val_subset = Subset(val_set, subset_indices_val_test)
    val_batches = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=4)
    test_subset = Subset(test_set, subset_indices_val_test)
    test_batches = DataLoader(test_subset, batch_size=4, shuffle=False, num_workers=4)

    # Setup model
    model = RGBUNet()

    # Load existing model
    # state_dict = torch.load("trained_models/rgb_unet_weights.pth", map_location='cpu')
    # model.load_state_dict(state_dict)

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
                if patience_counter >= patience_limit:
                    tqdm.write("Early stopping triggered.")
                    break

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving progress...")
        torch.save(model.state_dict(), f"{save_name}.pth")
        with open(f"{save_name}.json", 'w') as f:
            json.dump(history, f)


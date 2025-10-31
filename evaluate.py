import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from datasets.augmentations import get_image_net_normalization, get_rgb_transform, get_geometric_transform
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models import RGBUNet
from datasets import RGBDataset
from tqdm import tqdm


def plot_loss(loss_location, title):
    with open(loss_location, 'r') as f:
        data = json.load(f)
    plt.plot(data['train_loss'])
    plt.plot(data['val_loss'])
    plt.plot(data['val_dice'])
    plt.legend(['train_loss', 'val_loss', 'val_dice'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.show()


def show_predictions(model, dataloader, device, num_examples=4, class_colors=None):
    """
    Visualize input, ground truth, and predicted segmentation masks.

    Args:
        model: trained PyTorch segmentation model
        dataloader: DataLoader providing (images, labels)
        device: torch.device('cuda' or 'cpu')
        num_examples: number of samples to display
        class_colors: optional list or np.array of shape (num_classes, 3) for RGB colors
    """
    model.eval()
    shown = 0

    if class_colors is not None:
        class_colors = np.array([mcolors.to_rgb(c) for c in class_colors])

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Move to CPU for plotting
            images = images.cpu()
            labels = labels.cpu()
            preds = preds.cpu()

            for i in range(images.shape[0]):
                if shown >= num_examples:
                    return  # stop when enough examples are shown

                img = images[i].permute(1, 2, 0).numpy()
                gt = labels[i].numpy()
                pr = preds[i].numpy()

                # Normalize image for display
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                # Optional color mapping for segmentation masks
                if class_colors is not None:
                    gt_rgb = class_colors[gt]
                    pr_rgb = class_colors[pr]
                else:
                    gt_rgb = gt
                    pr_rgb = pr

                # Plot triplet
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title("Input Image")
                axes[1].imshow(gt_rgb)
                axes[1].set_title("Ground Truth")
                axes[2].imshow(pr_rgb)
                axes[2].set_title("Prediction")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()
                plt.show()
                shown += 1


def plot_confusion_matrix(cm, class_names):
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # replaces NaNs and infs with 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, xticklabels=class_names, yticklabels=class_names,
                annot=True, cmap='Blues', fmt=".2f")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Normalized Confusion Matrix (per true class)")
    plt.show()


def evaluate_per_class_accuracy(model, dataloader, device, class_names):
    model.eval()
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Move to CPU for NumPy
            y_true = labels.cpu().numpy().ravel()
            y_pred = preds.cpu().numpy().ravel()

            # Update confusion matrix
            cm += confusion_matrix(
                y_true, y_pred, labels=np.arange(num_classes)
            )

            del images, labels, outputs, preds
            torch.cuda.empty_cache()

    # Compute accuracies
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)
    mean_acc = np.mean(per_class_acc)

    plot_confusion_matrix(cm, class_names)
    return per_class_acc, mean_acc, cm


def create_datasets_splits(folders, splits, seed):
    datasets = []
    for folder in folders:
        datasets.append(RGBDataset(rgb_dir=f'{folder}/ortho/',
                             mask_dir=f'{folder}/brt/',
                             normalization=get_image_net_normalization()))
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
    model_weights = "C:/MTP-Code/trained_models/rgb_unet_varied_full.pth"
    loss_location = "C:/MTP-Code/trained_models/rgb_unet_varied_full.json"
    data_folders = ["C:/MTP-Data/dataset_diverse_2022_512/bies_bosch",
                    "C:/MTP-Data/dataset_diverse_2022_512/schoorl",
                    "C:/MTP-Data/dataset_diverse_2022_512/vierhouten",
                    "C:/MTP-Data/dataset_diverse_2022_512/soesterberg"]
    title = "full dataset, batch size 8"
    class_map = "C:/MTP-Code/Data/BRT/class_map.json"
    color_map = "C:/MTP-Code/Data/BRT/class_to_color.json"
    plot_examples = 20

    with open(class_map, 'r') as f:
        class_map = json.load(f)
        classes = [k for k, v in class_map.items()]

    with open(color_map, 'r') as f:
        class_map = json.load(f)
        colors = [v for k, v in class_map.items()]

    # Setup model
    model = RGBUNet()
    model.load_state_dict(torch.load(model_weights))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup dataloader
    train_set, val_set, test_set = create_datasets_splits(folders=data_folders, splits=[0.8, 0.1, 0.1], seed=43)

    # Subset train and test set
    test_batches = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=4)

    plot_loss(loss_location, title)
    per_class_acc, mean_acc, cm = evaluate_per_class_accuracy(model, test_batches, device, classes)
    show_predictions(model, test_batches, device, num_examples=plot_examples, class_colors=colors)

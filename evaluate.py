import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from datasets.augmentations import get_image_net_normalization, get_rgb_transform, get_geometric_transform
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models import RGBUNet, EarlyFusionUNet
from datasets import RGBDataset, RGBDSMDataset
from tqdm import tqdm
import os


def plot_loss(loss_location, title):
    with open(loss_location, 'r') as f:
        data = json.load(f)

    epochs = range(1, len(data['train_loss']) + 1)

    plt.plot(data['train_loss'])
    plt.plot(data['val_loss'])
    plt.plot(data['val_dice'])
    plt.legend(['train_loss', 'val_loss', 'val_dice'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(range(0, len(epochs) + 1, 5))
    plt.title(title)
    plt.show()


def show_predictions(model, dataloader, device, output_dir, title="", num_examples=4, class_colors=None):
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

    os.makedirs(output_dir, exist_ok=True)

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
                if images[i].shape[0] == 4:
                    rgb, _ = torch.split(images[i], [3, 1], dim=0)
                else:
                    rgb = images[i]
                img = rgb.permute(1, 2, 0).numpy()
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

                adjusted_title = f'{title}, example nr. {shown + 1}'
                fig.suptitle(adjusted_title, fontsize=16)
                output_path = os.path.join(output_dir, f"example_{shown + 1}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to free memory
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


if __name__ == '__main__':
    model_weights = "C:/MTP-Data/trained_models/rgbdsm_u_net_pr34_b32_m/rgbdsm_u_net_pr34_b32_m.pth"
    loss_location = "C:/MTP-Data/trained_models/rgbdsm_u_net_pr34_b32_m/rgbdsm_u_net_pr34_b32_m.json"
    image_folder = "C:/MTP-Data/trained_models/rgbdsm_u_net_pr34_b32_m/images"
    title = "RGBDSM: Pre-trained ResNet-34, batch size 32"

    rgb_folder = "C:/MTP-Data/dataset_diverse_2022_512_sep/test/ortho"
    dsm_folder = "C:/MTP-Data/dataset_diverse_2022_512_sep/test/dsm"
    mask_folder = "C:/MTP-Data/dataset_diverse_2022_512_sep/test/brt"
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
    model = EarlyFusionUNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_weights, weights_only=True, map_location=device))
    model.to(device)

    # Setup dataloader
    test_set = RGBDSMDataset(rgb_dir=rgb_folder,
                             mask_dir=mask_folder,
                             dsm_dir=dsm_folder,
                             normalization=get_image_net_normalization())

    # Subset train and test set
    test_batches = DataLoader(test_set, batch_size=2, shuffle=True, num_workers=4)

    # plot_loss(loss_location, title)
    # per_class_acc, mean_acc, cm = evaluate_per_class_accuracy(model, test_batches, device, classes)
    show_predictions(model, test_batches, device,
                     num_examples=plot_examples,
                     class_colors=colors,
                     output_dir=image_folder,
                     title=title)

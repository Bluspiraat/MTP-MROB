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


def plot_loss(loss_location, title, output_path):
    with open(loss_location, 'r') as f:
        data = json.load(f)

    best_epoch = data['val_loss'].index(min(data['val_loss']))
    best_epoch_val = data['val_loss'][best_epoch]

    epochs = range(1, len(data['train_loss']) + 1)

    plt.plot(range(1, len(epochs)+1), data['train_loss'])
    plt.plot(range(1, len(epochs)+1), data['val_loss'])
    plt.plot(range(1, len(epochs)+1), data['val_dice'])
    plt.legend(['train_loss', 'val_loss', 'val_dice'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(range(1, len(epochs)+1, 5))
    plt.title(title + f'\n Lowest validation loss is {best_epoch_val:.4f} at epoch: {best_epoch+1}')
    plt.tight_layout()
    plt.savefig(f'{output_path}/loss_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Close the figure to free memory


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

    os.makedirs(os.path.join(output_dir, 'examples'), exist_ok=True)

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

                adjusted_title = f'{title}, example nr. {shown + 1:02d}'
                fig.suptitle(adjusted_title, fontsize=16)
                output_path = os.path.join(output_dir, f"examples/example_{shown + 1}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to free memory
                shown += 1


def plot_confusion_matrix(cm, class_names, output_path):
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # replaces NaNs and infs with 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, xticklabels=class_names, yticklabels=class_names,
                annot=True, cmap='Blues', fmt=".2f")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Normalized Confusion Matrix (per true class)")
    plt.tight_layout()
    plt.savefig(f'{output_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Close the figure to free memory


def evaluate_per_class_accuracy(model, dataloader, device, class_names, output_path):
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

    plot_confusion_matrix(cm, class_names, output_path)
    return per_class_acc, mean_acc, cm

def segmentation_metrics_from_cm(cm, class_names=None, output_dir=None):
    cm = cm.astype(np.float64)

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    total = np.sum(cm)

    IoU = TP / (TP + FP + FN + 1e-12)
    mIoU = np.nanmean(IoU)

    freq = np.sum(cm, axis=1) / total
    FWIoU = np.sum(freq * IoU)

    Dice = 2 * TP / (2 * TP + FP + FN + 1e-12)
    mean_Dice = np.nanmean(Dice)

    PA = np.sum(TP) / total
    class_PA = TP / np.sum(cm, axis=1)
    mPA = np.nanmean(class_PA)

    # Build results dictionary
    results = {
        "Pixel_Accuracy": float(PA),
        "Mean_Pixel_Accuracy": float(mPA),
        "Mean_IoU": float(mIoU),
        "Frequency_Weighted_IoU": float(FWIoU),
        "Mean_Dice": float(mean_Dice),
        "Per_Class": {}
    }

    # Add per-class metrics
    for i in range(len(class_names)):
        name = class_names[i]
        results["Per_Class"][name] = {
            "IoU": float(IoU[i]),
            "Dice": float(Dice[i]),
            "Pixel_Accuracy": float(class_PA[i])
        }

    # Optionally save to JSON
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Metrics saved to {output_path}")


if __name__ == '__main__':
    model_weights = "C:/MTP-Data/trained_models/rgbdsm_u_net_r34_b16_a03/rgbdsm_u_net_r34_b16_a03.pth"
    loss_location = "C:/MTP-Data/trained_models/rgbdsm_u_net_r34_b16_a03/loss_values.json"
    folder = "C:/MTP-Data/trained_models/rgbdsm_u_net_r34_b16_a03/"
    title = "RGBDSM: Pre-trained ResNet-34, batch size 16, alpha 0.3"

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

    # Setup
    # model = RGBUNet(encoder_name='resnet34')
    model = EarlyFusionUNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_weights, weights_only=True, map_location=device))
    model.to(device)

    # Setup dataloader
    test_set = RGBDSMDataset(rgb_dir=rgb_folder,
                             mask_dir=mask_folder,
                             dsm_dir=dsm_folder,
                             normalization=get_image_net_normalization())
    # test_set = RGBDataset(rgb_dir=rgb_folder,
    #                       mask_dir=mask_folder,
    #                       normalization=get_image_net_normalization())

    test_batches = DataLoader(test_set, batch_size=4, shuffle=True, num_workers=4)

    plot_loss(loss_location, title, folder)
    per_class_acc, mean_acc, cm = evaluate_per_class_accuracy(model, test_batches, device, classes, folder)
    segmentation_metrics_from_cm(cm, output_dir=folder, class_names=classes)
    show_predictions(model, test_batches, device,
                     num_examples=plot_examples,
                     class_colors=colors,
                     output_dir=folder,
                     title=title)

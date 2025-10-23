import torch
import torch.nn.functional as F
from tqdm import tqdm
import json


def dice_score_multiclass(preds, targets, num_classes, eps=1e-6):
    preds = F.softmax(preds, dim=1)
    preds_classes = preds.argmax(dim=1)  # [B,H,W]
    dice_scores = []

    for cls in range(num_classes):
        pred_cls = (preds_classes == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum(dim=(1,2))
        union = pred_cls.sum(dim=(1,2)) + target_cls.sum(dim=(1,2))
        dice = (2 * intersection + eps) / (union + eps)
        dice_scores.append(dice.mean().item())

    return sum(dice_scores) / num_classes


def evaluate_model(model, dataloader, num_classes, device, class_map):
    model.eval()

    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for rgb, mask in tqdm(dataloader, desc="Evaluating", leave=False):
            rgb = rgb.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)

            # Predict
            logits = model(rgb)
            preds = torch.argmax(logits, dim=1)  # [B,H,W]

            # Flatten both
            preds = preds.view(-1)
            mask = mask.view(-1)

            # Update per-class counts
            for c in range(num_classes):
                class_mask = (mask == c)
                total_per_class[c] += class_mask.sum().cpu()
                correct_per_class[c] += (preds[class_mask] == c).sum().cpu()

    # Compute accuracy per class
    accuracy_per_class = correct_per_class.float() / total_per_class.float()
    accuracy_per_class = accuracy_per_class.cpu().numpy()

    with open(class_map, "r") as f:
        class_map = json.load(f)
        index_to_class = {v: k for k, v in class_map.items()}

    evaluation = {}
    for i, acc in enumerate(accuracy_per_class):
        if torch.isnan(torch.tensor(acc)):
            print(f"Class {index_to_class[i]}: N/A (not present in test set)")
            evaluation[i] = float('NaN')
        else:
            print(f"Class {index_to_class[i]}: {acc:.3f}")
            evaluation[i] = acc

    with open('evaluation.json', 'w') as f:
        json.dump(evaluation, f)

    return accuracy_per_class


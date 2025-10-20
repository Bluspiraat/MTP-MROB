from datasets import RGBDataset
from torch.utils.data import DataLoader, random_split, Subset
from models import RGBUNet
import torch
from tqdm import tqdm

if __name__ == '__main__':

    # Load data
    rgb_folder = "C:/MTP-Data/dataset_twente_512/ortho/"
    mask_folder = "C:/MTP-Data/dataset_twente_512/brt/"
    dataset = RGBDataset(rgb_folder, mask_folder)

    train_set, test_set = random_split(dataset, [0.8, 0.2])
    subset_indices = list(range(100))
    train_subset = Subset(train_set, subset_indices)
    train_batches = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=4)

    # Setup model
    model = RGBUNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Setup losses function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    predict_epochs = [1, 5, 10]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Wrap dataloader with tqdm
        loop = tqdm(train_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for rgb, mask in loop:
            rgb = rgb.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(rgb)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update tqdm description dynamically
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_batches)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed, Avg Loss: {avg_loss:.4f}")
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from dataset import PokemonsData
from models.model import PokemonClassifier

# --- Helper Functions ---
def train_step(model, batch, criterion, optimizer, device):
    """Single training step."""
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Full epoch training."""
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        running_loss += train_step(model, batch, criterion, optimizer, device) * len(batch[0])
    return running_loss / len(dataloader.dataset)

# --- Main Training Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)    
    # Model setup
    model = PokemonClassifier(num_classes=102).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Data setup
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = PokemonsData(csv_file="./data/pokemon.csv", 
                          root_dir="./data/pokemon_cards_gen1/", 
                          transform=transform)
    
    # Split dataset (example: 80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Training
    epochs = 14
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    
    torch.save(model.state_dict(), "saved_models/model.pth")

if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn

from dataset import PokemonsData
from models.model import PokemonClassifier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = PokemonClassifier(num_classes=102).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # <- this converts the PIL image to a torch.Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # optional, good for models
])

# Dataset and Dataloader
data_folder = "./data/pokemon_cards_gen1/"
csv_file = "./data/pokemon.csv"
pokemon_dataset = PokemonsData(csv_file=csv_file, root_dir=data_folder, transform=transform)
pokemon_dataloader = DataLoader(pokemon_dataset, batch_size=32, shuffle=True)

# Training
epochs = 14
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in pokemon_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(pokemon_dataloader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
torch.save(model.state_dict(), "saved_models/model.pth")

from PIL import Image
import torch
from torchvision import transforms
from models.model import PokemonClassifier
from classes import idx_to_class



model = PokemonClassifier()


model.load_state_dict(torch.load("saved_models/model.pth", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

img = Image.open("test_images/test.png").convert("RGB")
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    pred_idx = output.argmax(dim=1).item()

print("Prediction:", idx_to_class[pred_idx])


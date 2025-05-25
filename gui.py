import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from models.model import PokemonClassifier  # your model file

# Load model (make sure to load it on CPU or CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PokemonClassifier(num_classes=102)
model.load_state_dict(torch.load("saved_models/model.pth", map_location=device))
model.to(device)
model.eval()

# Transform to match training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Your class names, make sure these match your dataset labels
class_names = ["Bagon", "Blaziken", "Caterpie", "Golem", "Klink", 
               "Mr.Mime", "Rockruff", "Snorlax", "Wailmer", "Zorua"]

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        result_label.config(text="Predicting...")
        
        prediction = predict_image(file_path)
        result_label.config(text=f"Prediction: {prediction}")

root = tk.Tk()
root.title("Pokemon Classifier")

load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack()

root.mainloop()


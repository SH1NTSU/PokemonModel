from models.model import PokemonClassifier
import torch
from dataset import PokemonsData
from torchvision import transforms
from torch.utils.data import DataLoader

def test_dataloader():
    root_dir = "./data/pokemon_cards_gen1/"
    csv_file = "./data/pokemon.csv" 
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # <- this converts the PIL image to a torch.Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # optional, good for models
    ])
    data = PokemonsData(root_dir=root_dir, csv_file=csv_file, transform=transform)
    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    batch = next(iter(dataloader))

    assert batch[0].shape == (32, 3, 128, 128); "wrong shape!"
    




def test_model():
    test_input = torch.rand(2, 3, 224, 224)
    model = PokemonClassifier()
    test_output = model(test_input)
    assert test_output.shape == (2, 102)
    




# def test_training():
#     model = PokemonClassifier()
#     optimizer = torch.optim.Adam(model.parameters())
#     loss_fn = torch.nn.CrossEntropyLoss()
#     
#     x = torch.randn(1, 3, 224, 224)
#     y = torch.tensor([1])
#     
#     loss = train_step(model, x, y, optimizer, loss_fn)
#     assert not torch.isnan(loss), "Loss is NaN
#
#

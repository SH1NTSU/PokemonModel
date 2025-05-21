import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from dataset import PokemonsData


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



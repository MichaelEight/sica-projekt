import torch
from model.models import Inception1DNet # Twoja klasa modelu

# Zainicjuj model
model = Inception1DNet(num_classes=8)
model.eval()

# Stwórz sztuczny tensor o wymiarach wejścia (Batch, Kanały, Długość)
dummy_input = torch.randn(1, 12, 5000)

# Wyeksportuj model do pliku
torch.onnx.export(model, dummy_input, "inception1d.onnx")
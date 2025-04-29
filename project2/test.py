import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pt"  # Your trained model file

CLASS_NAMES = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
    'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
    'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
    'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
    'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
    'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier'
]

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Model Definition (must match training) ---
class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, output_features)

    def forward(self, x):
        return self.model(x)

# --- Load Model ---
def load_model():
    model = PretrainedModel(output_features=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --- Predict Image ---
def predict_image(img_path):
    model = load_model()

    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_breed = CLASS_NAMES[predicted_idx.item()]

    print(f"🔮 Prediction: {predicted_breed}")

if __name__ == "__main__":
    predict_image("download.png")

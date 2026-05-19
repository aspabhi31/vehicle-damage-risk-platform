import os
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# ============================================================
# MODEL PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "damage_classifier.pth"
)

# ============================================================
# CLASS LABELS
# ============================================================
CLASS_NAMES = [
    "Front Breakage",
    "Front Crushed",
    "Front Normal",
    "Rear Breakage",
    "Rear Crushed",
    "Rear Normal"
]

# ============================================================
# MODEL
# ============================================================
class CarClassifierCNN(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        self.model = models.resnet50(weights=None)

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):

        return self.model(x)

# ============================================================
# LOAD MODEL
# ============================================================
trained_model = CarClassifierCNN()

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}"
    )

trained_model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)

trained_model.eval()

# ============================================================
# TRANSFORMS
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# PREDICTION
# ============================================================
def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    x = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = trained_model(x)

        probs = torch.softmax(outputs, dim=1)

        pred_idx = torch.argmax(
            probs,
            dim=1
        ).item()

    return CLASS_NAMES[pred_idx]

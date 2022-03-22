import os
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader
from model import Classifier
from Dataset import FoodDataset


test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_dataset_dir = "../food11"
test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

"""# Testing and generate prediction CSV"""

model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"train_best.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in test_loader:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)

df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1, len(test_set)+1)]
df["Category"] = prediction
df.to_csv("submission.csv", index = False)
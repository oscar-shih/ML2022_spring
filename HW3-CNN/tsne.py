import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

import torch.nn as nn
from torchvision import models

from Dataset import *
from model import Classifier

batch_size = 128
_dataset_dir = "../food11"
valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=train_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Classifier().to(device)
model.load_state_dict(torch.load(f'train_best.ckpt'))
model.eval()

output_name, raw_prediction, output = [], [], []
features = []
labels = []

for batch in tqdm(valid_loader):
    imgs, lbls = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
    labels.extend(lbls.cpu().numpy())
    logits = np.squeeze(logits.cpu().numpy())
    features.extend(logits)
    
colors_per_class = [list(np.random.choice(range(256), size=3)) for i in range(11)]
# print(labels[0], features[0])
tsne = TSNE(n_components=2, init='pca').fit_transform(features)

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)

    return starts_from_zero / value_range

def visualize_tsne_points(tx, ty, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for label in range(11):
        indices = [i for i, l in enumerate(labels) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255
        ax.scatter(current_tx, current_ty, c=color, label=label, s=5)

    ax.legend(loc='best')
    plt.show()

def visualize_tsne(tsne, labels, plot_size=1000, max_image_size=100):
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    visualize_tsne_points(tx, ty, labels)

from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
visualize_tsne(tsne, labels)
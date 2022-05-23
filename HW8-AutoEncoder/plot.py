import random
import enum
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(48763)

class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(), 
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 64 * 64 * 3), 
            nn.Tanh()
        )

    def forward(self, x, latent):
        x = self.encoder(x)
        x += latent
        x = self.decoder(x)
        return x

class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        
        self.transform = transforms.Compose([
          transforms.Lambda(lambda x: x.to(torch.float32)),
          transforms.Lambda(lambda x: 2. * x/255. - 1.),
        ])
        
    def __getitem__(self, index):
        x = self.tensors[index]
        
        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)

train = np.load('data/trainingset.npy', allow_pickle=True)
eval_batch_size = 128
model_type = 'fcn'

x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=128)

data = next(iter(train_dataloader))
img = data.float().cuda()
img = img.view(img.shape[0], -1)
latent = torch.zeros((eval_batch_size, 256)).cuda()

# load trained model
checkpoint_path = f'last_model_{model_type}.pt'
model = torch.load(checkpoint_path)
model.eval()

# Model Architecture
with open("./model.txt", "w") as out:
    out.write(str(model))

# Original image
origin = img[0].detach().cpu().numpy().reshape(3, 64, 64).transpose(1, 2, 0) + 1
origin = (origin * 255. / 2.).astype(int)
plt.imshow(origin)
plt.savefig("origin.png")

# Model Output
output = model(img, latent)
output_origin = output[0].detach().cpu().numpy().reshape(3, 64, 64).transpose(1, 2, 0) + 1
output_origin = (output_origin * 255. / 2.).astype(int)
plt.imshow(output_origin)
plt.savefig("output.png")

# Transform 1
latent[0] += 1.5
# latent[1] += 1.5
output_t1 = model(img, latent)
output_t1_img = output_t1[0].detach().cpu().numpy().reshape(3, 64, 64).transpose(1, 2, 0) + 1
output_t1_img = (output_t1_img * 255. / 2.).astype(int)
plt.imshow(output_t1_img)
plt.savefig("output_t1.png")

# Transform 2
latent[0] -= 2
output_t2 = model(img, latent)
output_t2_img = output_t2[0].detach().cpu().numpy().reshape(3, 64, 64).transpose(1, 2, 0) + 1
output_t2_img = (output_t2_img * 255. / 2.).astype(int)
plt.imshow(output_t2_img)
plt.savefig("output_t2.png")
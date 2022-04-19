import os
from util import same_seeds
from data import get_dataset
from train import TrainerGAN

if __name__ == "__main__":
    same_seeds(1126)
    workspace_dir = '.'
    temp_dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
    config = {
        "model_type": "GAN",
        "batch_size": 64,
        "lr": 1e-4,
        "n_epoch": 1,
        "clip_value": 0.001,
        "n_critic": 1,
        "z_dim": 100,
        "workspace_dir": workspace_dir, # define in the environment setting
    }

    trainer = TrainerGAN(config)
    trainer.train()

    # save the 1000 images into ./output folder
    trainer.inference(f'{workspace_dir}/checkpoints/') # you have to modify the path when running this line

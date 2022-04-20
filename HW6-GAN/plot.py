# Report Q2 Plot Generation -- Gradient Norm v.s Convolution Layers from 1 to 5
import os
import numpy as np
import matplotlib.pyplot as plt
from model import Discriminator
from train import TrainerGAN

if __name__ == "__main__":
    # TODO: Plot the required plot for Q2.
    wc, gp = [], []
    with open("./WGAN.txt") as wgan:
        for lines in wgan:
            wc.append(float(lines.split("\n")[0]))

    with open("./WGAN_GP.txt") as wgan_gp:
        for lines in wgan_gp:
            gp.append(float(lines.split("\n")[0]))
    wc.reverse()
    gp.reverse()
    
    layer = [5, 4, 3, 2, 1]
    plt.figure(figsize=(10, 10), linewidth=2)
    plt.xticks(layer)
    plt.plot(layer, wc, 's-', color='r', label="Weight Clipping(c=0.1)")
    plt.plot(layer, gp, 'o-', color='g', label="Gradient Penalty")
    plt.axis([max(layer)+1, min(layer)-1, min(gp)-1, max(wc)+1])
    plt.xlabel("Discriminator Layer", fontsize=30, labelpad=15)
    plt.ylabel("Gradient norm (log scale)", fontsize=30, labelpad=15)
    plt.legend(loc="best", fontsize=15)
    # plt.show()
    plt.savefig("Q2.png")
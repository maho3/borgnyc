# %load_ext autoreload
# %autoreload 2

import numpy as np
# # import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import join
import os
from PIL import Image
from tqdm import tqdm

trial = 0
wdir = '/ocean/projects/phy240015p/mho1/borgnyc'
odir = join(wdir, f'trial{trial}')

os.listdir(odir)

f = plt.figure(figsize=(40, 27), facecolor='black')
gs = mpl.gridspec.GridSpec(4, 6, hspace=0.05, wspace=0.05)

for i in range(8):
    ax = plt.subplot(gs[i % 4, i//4])
    r = np.load(join(odir, f'dens_{8*i}.npy'))
    p = 64
    r = r[p:-p, p:-p]
    ax.imshow(r, cmap='inferno', vmin=0, vmax=30, interpolation='spline16')
    ax.axis('off')

rho = np.load(join(odir, 'rho.npy'))
ax = plt.subplot(gs[:, 2:])
r = rho[p:-p, p:-p]
ax.imshow(r, cmap='inferno', vmin=0, vmax=30, interpolation='spline16')
ax.axis('off')

plt.savefig('densities.png', dpi=72, bbox_inches='tight', facecolor='black')

import os

# # change JAX GPU memory preallocation fraction
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'  # noqa

import jax
from tqdm import tqdm

import jax.numpy as jnp

from pmwd import (
    Configuration,
    SimpleLCDM,
    boltzmann,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)

from PIL import Image
from os.path import join

import matplotlib.pyplot as plt
from functools import partial

from jax.example_libraries.optimizers import adam
import numpy as np
import time

# Check if JAX is using GPU
from jax import default_backend
print('Jax is using', default_backend())

# ~~~ SET UP CONFIGURATION ~~~
trial = 0
seed = 5423
# targetdim = (32, 32)
targetdim = (768, 768)
# targetdim = (512, 512)
depth = 8
ptcl_spacing = 6
Niters = 200
mesh_shape = 1

wdir = '/ocean/projects/phy240015p/mho1/borgnyc'
odir = join(wdir, f'trial{trial}')
os.makedirs(odir, exist_ok=True)

t0 = time.time()


# ~~~ SET UP IMAGE ~~~
print('Setting up IMAGE...')
# Load the image
image_path = 'iss045e066112.jpeg'
image = Image.open(image_path)

# Convert the image to grayscale
image = image.convert('L')

# rotate
ang = -135
image = image.rotate(ang)

# crop the picture
xmin, ymin = 350, 50
xmax, ymax = xmin+512, ymin+512
image = image.crop((xmin, ymin, xmax, ymax))

# add black space to the edges
new_im = Image.new("L", (640, 640))
new_im.paste(image, (64, 64))
image = new_im

# downscale
image = image.resize(targetdim)

# Convert the image to a numpy array
image = jnp.array(image)

# Save the outputs
f, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.axis('off')
f.savefig(join(odir, 'truth.jpg'), bbox_inches='tight', pad_inches=0)
plt.close(f)


# ~~~ SET UP PMWD ~~~
print('Setting up PMWD...')
ptcl_grid_shape = (*targetdim, depth)

# normalize the image to make the target
im_tgt = image / 255
im_tgt *= jnp.prod(jnp.array(ptcl_grid_shape)) / im_tgt.sum()

# set up grid
conf = Configuration(
    ptcl_spacing, ptcl_grid_shape, mesh_shape=mesh_shape)
cosmo = SimpleLCDM(conf)
seed = 0
modes = white_noise(seed, conf, real=True)
cosmo = boltzmann(cosmo, conf)
print(conf)


# ~~~ SET UP FUNCTIONS ~~~
print('Setting up FUNCTIONS...')


def model(modes, cosmo, conf):
    modes = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    dens = scatter_2d(ptcl, conf)
    return ptcl, dens


def scatter_2d(ptcl, conf):
    dens = jnp.zeros(
        tuple(s//mesh_shape for s in conf.mesh_shape), dtype=conf.float_dtype)
    dens = scatter(ptcl, conf, mesh=dens, val=1, cell_size=conf.cell_size)
    return dens.sum(axis=2)


def obj(tgt, modes, cosmo, conf):
    _, dens = model(modes, cosmo, conf)
    return (dens - tgt).var() / tgt.var()


obj_valgrad = jax.value_and_grad(obj, argnums=1)


def optim(tgt, modes, cosmo, conf, iters=100, lr=0.1):
    init, update, get_params = adam(lr)
    state = init(modes)

    def step(i, state, tgt, cosmo, conf):
        modes = get_params(state)
        value, grads = obj_valgrad(tgt, modes, cosmo, conf)
        state = update(i, grads, state)
        return value, state

    tgt = jnp.asarray(tgt)
    for i in tqdm(range(iters)):
        value, state = step(i, state, tgt, cosmo, conf)

    modes = get_params(state)
    return value, modes


# ~~~ OPTIMIZE ~~~
print('Optimizing...')
loss, modes_optim = optim(im_tgt, modes, cosmo, conf, iters=Niters)
print(loss, modes.std(), modes_optim.std())

# ~~~ SAVE ~~~
print('Saving...')
# Save the optimized modes
modes_optim_path = join(odir, 'modes_optim.npy')
np.save(modes_optim_path, np.array(modes_optim))

# Save the particles
ptcl, rho = model(modes_optim, cosmo, conf)
ppos = ptcl.pos()
pvel = ptcl.vel
np.save(join(odir, 'ppos.npy'), np.array(ppos))
np.save(join(odir, 'pvel.npy'), np.array(pvel))

# Save the optimized density
rho_path = join(odir, 'rho.npy')
np.save(rho_path, np.array(rho))

f, ax = plt.subplots()
ax.imshow(rho, cmap='gray')
ax.axis('off')
f.savefig(join(odir, 'rho.jpg'), bbox_inches='tight', pad_inches=0)


print('Done in', (time.time() - t0)/60, 'm')

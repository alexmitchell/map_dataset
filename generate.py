
import torch
import numpy as np
import os

from train import model, generate
from diffusers.utils import make_image_grid, numpy_to_pil
from pathlib import Path

ckpt = torch.load('lightning_logs/version_20/checkpoints/epoch=99-step=204800.ckpt')
model.load_state_dict(ckpt['state_dict'])

rng = np.random.default_rng()
land_fraction = 0.5

images = generate(
        rng,
        model.model,
        model.scheduler,
        n=16,
        label=1000 * land_fraction,
)
images = numpy_to_pil(images.cpu().permute(0, 2, 3, 1).numpy())
image_grid = make_image_grid(images, rows=4, cols=4)

image_dir = Path('generate')
os.makedirs(image_dir, exist_ok=True)
image_grid.save(image_dir / f'epoch_99_land_{int(1000 * land_fraction)}.png')


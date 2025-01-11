import lightning as L
import torch
import numpy as np
import os

from diffusers import DDPMScheduler
from diffusers.utils import make_image_grid, numpy_to_pil
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import unet
from landcover_training_dataset import BinaryLandcoverDataset
from torchmetrics.image import TotalVariation
from pathlib import Path
from tqdm import tqdm

from typing import Optional

class DiffusionTask(L.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.optimizer = Adam(model.parameters())
        self.loss = MSELoss()
        self.tv = TotalVariation(reduction='mean')

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        x_clean, noise, t, label = x
        n = self.scheduler.config.num_train_timesteps

        # The model expects images to be 32-bit floats in [0, 1].
        x_clean = x_clean.to(dtype=torch.float32)
        noise = noise.to(dtype=torch.float32)

        # The dataset gives us the timestep as a float in [0, 1), so we have to 
        # scale it up and convert it to an integer.
        t = (t * n).to(dtype=int)

        # Make the land fraction label comparable in scale to the timestep, 
        # since they're both embedded in the same way.
        label *= n

        x_noisy = self.scheduler.add_noise(x_clean, noise, t)

        noise_pred = self.model(x_noisy, t, label).sample

        return self.loss(noise_pred, noise)

    def training_step(self, batch, _):
        loss = self.forward(batch)
        self.log('train/loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss = self.forward(batch)
        self.log('val/loss', loss)
        return loss

    def test_step(self, batch, _):
        loss = self.forward(batch)
        self.log('test/loss', loss)
        return loss

    def on_validation_epoch_end(self):
        rng = np.random.default_rng(0)
        images = generate(
                rng=rng,
                model=self.model,
                scheduler=self.scheduler,
                n=1024,
                label=500,
        )

        self.tv.update(images)
        self.log('gen/tv', self.tv.compute())
        self.tv.reset()

        images = numpy_to_pil(images[:16].cpu().permute(0, 2, 3, 1).numpy())
        image_grid = make_image_grid(images, rows=4, cols=4)

        if self.trainer.logger.log_dir:
            image_dir = Path(self.trainer.logger.log_dir) / 'gen_images'
            os.makedirs(image_dir, exist_ok=True)
            image_grid.save(image_dir / f'epoch_{self.current_epoch:02}.png')


class MapData(L.LightningDataModule):

    def __init__(
            self,
            *,
            resolution_px: int,
            train_epoch_size: int,
            val_epoch_size: int,
            batch_size: int,
            num_workers: Optional[int] = None,
    ):
        super().__init__()

        epoch_sizes = {
                'train': train_epoch_size,
                'val': val_epoch_size,
        }

        def make_dataloader(split):
            root = Path('data')

            # We should use different maps for training and validation, but 
            # right now we only have one.
            dataset = BinaryLandcoverDataset(
                    aoi_filepath=root/'greece'/'aoi.geojson',
                    projected_crs='EPSG:2100',
                    landcover_polygons_filepath=root/'landcover_100m_binary_vector.geojson',
                    tile_width_range_m=(1e2, 1e5),
                    tile_raster_size_px=resolution_px,
                    epoch_size=epoch_sizes[split],
            )

            return DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=get_num_workers(num_workers),
            )

        self._dataloaders = {
                k: make_dataloader(k)
                for k in epoch_sizes.keys()
        }

    def train_dataloader(self):
        return self._dataloaders['train']

    def val_dataloader(self):
        return self._dataloaders['val']

@torch.no_grad()
def generate(rng, model, scheduler, n, label):
    device = next(model.parameters()).device

    # Assume that the model is in eval-mode.  During training, lightning puts 
    # the model in eval-mode for validation.

    w = h = model.config.sample_size

    x = rng.normal(size=(n, 1, w, h))
    x = torch.from_numpy(x).to(device=device, dtype=torch.float32)

    label = torch.full((n,), label).to(device=device, dtype=torch.float32)

    for t in tqdm(scheduler.timesteps, desc='Generate images'):
        y = model(x, t, label).sample
        x = scheduler.step(y, t, x).prev_sample

    return x

def get_num_workers(num_workers: Optional[int]) -> int:
    if num_workers is not None:
        return num_workers
    try:
        return int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    except KeyError:
        return os.cpu_count()

torch.set_float32_matmul_precision('high')

model = DiffusionTask(unet)
data = MapData(
        resolution_px=32,
        train_epoch_size=2**17,
        val_epoch_size=2**14,
        batch_size=64,
)

if __name__ == '__main__':
    trainer = L.Trainer(
            max_epochs=100,
            #fast_dev_run=True,
    )
    trainer.fit(model, data)

from diffusers import UNet2DModel

unet = UNet2DModel(
        sample_size=32,
        in_channels=1,
        out_channels=1,

        # These parameters define a relatively simple model: not many layers, 
        # no attention, and small latent representations.  Our data is pretty 
        # simple, so I think this might work, but we have plenty of room to 
        # make the model more complicated as necessary.
        down_block_types=['DownBlock2D', 'DownBlock2D'],
        up_block_types=['UpBlock2D', 'UpBlock2D'],
        block_out_channels=[16, 32],
        norm_num_groups=8,

        # Our class labels give the fraction of the map that is land.  This is 
        # conceptually similar to the timestep; e.g. it's a floating point 
        # value that varies between a minimum and a maximum.  So I think it 
        # makes sense to embed the class label in the same way as the timestep: 
        # with a sinusoidal embedding.  This will probably work best if the 
        # land fraction is scaled to approximately the same range as the 
        # timesteps.  Note that the other embedding options are a lookup table 
        # or an identity function, and neither of those seem appropriate.
        class_embed_type='timestep',
)

if __name__ == '__main__':
    import torch
    x = torch.randn(1, 1, 32, 32)
    t = torch.randn(1)
    y = torch.randn(1)

    unet(x, t, y)

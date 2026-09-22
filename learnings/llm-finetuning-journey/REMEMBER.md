
> zero-grad

> data-loader reads from file everytime using, then applies transformation and then sends data for processing when num_workers=0. When you set num_workers=4 (or higher), PyTorch spawns 4 separate background CPU processes. These workers constantly read from the disk and apply transforms in the background, filling up a queue with ready-to-go batches in RAM. When the GPU finishes a batch, the next batch is instantly pulled from RAM, masking the slow disk read times.

> pre-process input for already trained models with that model's preprocessing parameters

## model

`nn.Sequential` containing `nn.modules` 

**CNN** usually will have `.features` (which derives the featuremap) and `.classifier` for the classification usecase models


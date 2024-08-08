# P-Hologen: An End-to-End Generative Framework for Phase-Only Holograms

This is the codebase for the paper P-Hologen: An End-to-end Generative Framework for Phase-Only Holograms.



## 0. Environment setup

Run the below command to prepare the virtual environment.

`conda env create --file environment.yaml`

`conda activate p_hologen`



Training on CelebA-HQ dataset requires the dataset to be prepared beforehand with the dataset directory configured as follows:



- datasets/celeba_hq
  - train
    - male
      - 0.png
      - 1.png
      - ...
    - female
      - 0.png
      - 1.png
      - ...
  - val
    - male
      - 0.png
      - 1.png
      - ...
    - female
      - 0.png
      - 1.png
      - ...



## 1. Training P-Hologen



- MNIST 64x64

`python train_p_hologen.py --dataset_name MNIST --image_size 64 --in_channels 1 --mse_w 0.9 --percept_w 0.1 --batch_size 50 --prop_dist "15.0_mm" --feature_size 6.4 `



- CelebA-HQ 128x128

`python train_p_hologen.py --dataset_name Celeba_HQ --image_size 128 --in_channels 3 --mse_w 0.9 --percept_w 0.1 --batch_size 20 --prop_dist "21.5_mm" --feature_size 6.4 `



The checkpoint files will be saved in `./ckpt_outputs/p_hologen/`.





## 2. Training PixelSnail



- MNIST 64x64

`python train_pixelsnail.py --dataset_name MNIST --image_size 64 --in_channels 1 --mse_w 0.9 --percept_w 0.1 --batch_size 50 --prop_dist "15.0_mm" --feature_size 6.4 --num_epochs 300`



- CelebA-HQ 128x128

`python train_pixelsnail.py --dataset_name Celeba_HQ --image_size 128 --in_channels 3 --mse_w 0.9 --percept_w 0.1 --batch_size 20 --prop_dist "21.5_mm" --feature_size 6.4 --num_epochs 300`



The checkpoint files will be saved in `./ckpt_outputs/PixSnail/`.





## 3. Evaluation

Evaluating the trained model can be done by adding various options to the below command:

`python evaluate.py --dataset_name <dataset_name> --image_size <image_size> --in_channels <number_of_input_channels> --mse_w <mse_weight> --percept_w <perceptual_weight> --prop_dist "<prop_dist>" --feature_size <feature_size> `



- Reconstruct a batch of validation images.

`--make_recons --batch_size <batch_size> `



- Generate samples.

`--gen_samples --num_samples <number_of_samples>`



- Evaluate reconstruction on the validation set images using psnr and ssim metrics.

`--eval_recons`



- Evaluate a batch of samples with a batch of validation set images using fid metric.

`--eval_samples --batch_size <batch_size>`



For example, generating 100 samples on MNIST can be done as follows:

`python evaluate.py --dataset_name MNIST --image_size 64 --in_channels 1 --mse_w 0.9 --percept_w 0.1 --prop_dist "15.0_mm" --feature_size 6.4 --gen_samples --num_samples 100 `



The generated results from the evaluation will be saved in `./training_results/samples/`.


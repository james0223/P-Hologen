import torch
from tqdm import tqdm
from modules.p_hologen import P_Hologen
import torchvision.transforms
from torch.utils.data import DataLoader
from pathlib import Path
import math

import torchvision.models as models
import torch.nn.functional as F

import argparse
import sys


def vgg_feats(vgg16, target_vgg_layers, x):
    features = {}
    for name, layer in vgg16._modules.items():
        x = layer(x)
        if name in target_vgg_layers:
            features[name] = x
    return features


def train_p_hologen(opts):
    dataset_name = opts.dataset_name
    input_channels = opts.in_channels
    img_size = opts.image_size
    if img_size == 64:
        ch_mult = [1, 2, 4]
        attn_resolutions = [16]
    elif img_size == 128:
        ch_mult = [1, 2, 3, 4]
        attn_resolutions = [16]
    elif img_size == 256:
        ch_mult = [1, 1, 2, 2, 4]
        attn_resolutions = [16]
    elif img_size == 512:
        ch_mult = [1, 1, 1, 2, 2, 4]
        attn_resolutions = [16]
    elif img_size == 1024:
        ch_mult = [1, 1, 1, 2, 2, 2, 4]
        attn_resolutions = [16]
    else:
        raise AssertionError("Inappropriate image size")

    str_ch_mult = [str(x) for x in ch_mult]
    txt_for_save = '_'.join(str_ch_mult)

    prop_val, prop_metric = opts.prop_dist.split("_")
    prop_val = float(prop_val)
    prop_metric = metric_dict[prop_metric]
    prop_dist = (prop_val * prop_metric, prop_val * prop_metric, prop_val * prop_metric)

    feat_val = float(opts.feature_size)
    feature_size = (feat_val * um, feat_val * um)

    z_dim = opts.z_dim
    k = opts.k
    device = opts.device
    print("Running on device: {}".format(device))

    num_epochs = opts.num_epochs
    batch_size = opts.batch_size
    lr = opts.lr
    result_save_every = opts.save_every
    commit_w = opts.commit_w
    scale_factor = opts.scale_factor
    max_patience = 15

    mse_w = opts.mse_w
    percept_w = opts.percept_w

    setting_memo = "p_hologen_{}_{}ch_z_dim{}_recon_w_m{}_p{}_chmult_{}_prop_{}_pixel_{}".format(
        dataset_name,
        input_channels,
        z_dim,
        mse_w,
        percept_w,
        txt_for_save,
        opts.prop_dist,
        opts.feature_size
    )

    print("setting memo: {}".format(setting_memo))

    result_output_path = "training_results/p_hologen/{}/{}".format(img_size, setting_memo)
    ckpt_output_path = "ckpt_outputs/p_hologen/{}/{}".format(img_size, setting_memo)
    Path(result_output_path).mkdir(parents=True, exist_ok=True)
    Path(ckpt_output_path).mkdir(parents=True, exist_ok=True)

    phase_transform = torchvision.transforms.ToPILImage()

    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ])

    if dataset_name == MNIST:

        trainset = torchvision.datasets.MNIST(root='datasets',
                                              train=True,
                                              transform=img_transform,
                                              download=True)

        valset = torchvision.datasets.MNIST(root='datasets',
                                            train=False,
                                            transform=img_transform,
                                            download=True)

    elif dataset_name == CELEBA_HQ:

        trainset = torchvision.datasets.ImageFolder('datasets/celeba_hq/train', transform=img_transform)

        valset = torchvision.datasets.ImageFolder('datasets/celeba_hq/val', transform=img_transform)

    else:
        raise AssertionError(
            "Invalid dataset name: {}, available: MNIST, Celeba_HQ, AFHQ".format(dataset_name))

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = P_Hologen(
        input_channels=input_channels,
        z_dim=z_dim,
        K=k,
        ch_mult=ch_mult,
        attn_resolutions=attn_resolutions,

        # Hologram
        img_size=img_size,
        feature_size=feature_size,
        wavelength_list=wavelength_list,
        distance_list=prop_dist,
        scale_output=scale_factor,
        pad_type="zero",

        # etc
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if percept_w > 0:
        vgg16 = models.vgg16(pretrained=True).features.eval().to(device)
        for param in vgg16.parameters():
            param.requires_grad = False

        target_vgg_layers = list(opts.vgg_layers.split("_"))  # ["8", "22"] # ReLU 2_2 , ReLU 4_3

        perceptual_scale = 1 / len(target_vgg_layers)

        print("Perceptual loss applied: Using VGG16 layers of {}".format(target_vgg_layers))
        print("Perceptual loss scaler: {}".format(perceptual_scale))
    else:
        print("Using Pixel MSE loss only")

    best_loss = sys.maxsize

    patience = 0

    for epoch in tqdm(range(num_epochs)):

        # Train
        model.train()
        for input_imgs, labels in train_dataloader:

            optimizer.zero_grad()

            B, C, H, W = input_imgs.shape

            if input_channels == 1 and C == 3:
                input_imgs = input_imgs[:, 1:2, :, :]

            input_imgs = input_imgs.to(device)

            poh, recon_img, z_e_x, z_q_x = model(input_imgs)

            # Reconstruction objective
            phase_recon_loss = 0

            if percept_w > 0:
                perceptual_loss = 0
                B, C, H, W = input_imgs.shape

                if C == 1:
                    vgg_input_recon_img = recon_img.expand(B, 3, H, W)
                    vgg_input_gt = input_imgs.expand(B, 3, H, W)
                else:
                    vgg_input_recon_img = recon_img
                    vgg_input_gt = input_imgs

                out_feats = vgg_feats(vgg16=vgg16, target_vgg_layers=target_vgg_layers, x=vgg_input_recon_img)
                gt_feats = vgg_feats(vgg16=vgg16, target_vgg_layers=target_vgg_layers, x=vgg_input_gt)

                for layer_name in target_vgg_layers:
                    perceptual_loss += perceptual_scale * F.mse_loss(out_feats[layer_name], gt_feats[layer_name])

                phase_recon_loss += percept_w * perceptual_loss

            if mse_w > 0:
                pixel_mse_loss = F.mse_loss(recon_img, input_imgs)
                phase_recon_loss += mse_w * pixel_mse_loss

            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())

            # Commitment objective
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

            loss = phase_recon_loss + loss_vq + commit_w * loss_commit

            loss.backward()

            optimizer.step()



        # Validation
        val_recon_phs_mse_loss = 0
        val_loss_vq = 0
        val_loss_commit = 0

        val_cnt = 0

        model.eval()
        with torch.no_grad():
            for input_imgs, labels in val_dataloader:

                B, C, H, W = input_imgs.shape

                if input_channels == 1 and C == 3:
                    input_imgs = input_imgs[:, 1:2, :, :]

                input_imgs = input_imgs.to(device)

                poh, recon_img, z_e_x, z_q_x = model(input_imgs)

                # Reconstruction objective
                phase_recon_loss = 0

                if percept_w > 0:
                    perceptual_loss = 0
                    B, C, H, W = input_imgs.shape

                    if C == 1:
                        vgg_input_recon_img = recon_img.expand(B, 3, H, W)
                        vgg_input_gt = input_imgs.expand(B, 3, H, W)
                    else:
                        vgg_input_recon_img = recon_img
                        vgg_input_gt = input_imgs

                    out_feats = vgg_feats(vgg16=vgg16, target_vgg_layers=target_vgg_layers, x=vgg_input_recon_img)
                    gt_feats = vgg_feats(vgg16=vgg16, target_vgg_layers=target_vgg_layers, x=vgg_input_gt)

                    for layer_name in target_vgg_layers:
                        perceptual_loss += perceptual_scale * F.mse_loss(out_feats[layer_name], gt_feats[layer_name])

                    phase_recon_loss += percept_w * perceptual_loss

                if mse_w > 0:
                    pixel_mse_loss = F.mse_loss(recon_img, input_imgs)
                    phase_recon_loss += mse_w * pixel_mse_loss

                # Vector quantization objective
                loss_vq = F.mse_loss(z_q_x, z_e_x.detach())

                # Commitment objective
                loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

                val_recon_phs_mse_loss += phase_recon_loss
                val_loss_vq += loss_vq
                val_loss_commit += commit_w * loss_commit

                val_cnt += 1

            val_avg_phs_mse = val_recon_phs_mse_loss / val_cnt
            val_avg_loss_vq = val_loss_vq / val_cnt
            val_avg_loss_commit = val_loss_commit / val_cnt

            avg_val_loss = val_avg_phs_mse + val_avg_loss_vq + val_avg_loss_commit

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience = 0
                torch.save(model.state_dict(), "{}/best_model.pth".format(ckpt_output_path))
                print("Average validation loss: {}".format(avg_val_loss))
                print("Saved best model for epoch {}".format(epoch))
                print("Patience reset to 0")
            else:
                patience += 1
                print("No improvement for epoch {}, patience: {}".format(epoch, patience))

            if not ((epoch + 1) % result_save_every) or (epoch + 1) == num_epochs:

                torchvision.utils.save_image(input_imgs[0], result_output_path + "/ep{}_original.png".format(epoch))
                torchvision.utils.save_image(recon_img[0], result_output_path + "/ep{}_recon.png".format(epoch))

                if input_channels == 1:
                    normalized_poh = phase_transform((poh[0].clone() + math.pi) / (2 * math.pi))
                    normalized_poh.save(result_output_path + "/ep{}_poh.png".format(epoch))
                else:
                    for poh_idx in range(3):
                        normalized_poh = phase_transform((poh[0][poh_idx].clone() + math.pi) / (2 * math.pi))
                        normalized_poh.save(result_output_path + "/ep{}_poh_{}channel.png".format(epoch, poh_idx))

        if patience >= max_patience:
            print("Out of patience, training process terminated on epoch {}".format(epoch))
            break
        else:
            print("Epoch {} finished!".format(epoch))

    print("Training finished!")
    return


if __name__ == "__main__":

    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

    metric_dict = {
        "cm": 1e-2,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
    }

    wavelength_list = (638 * nm, 520 * nm, 450 * nm)

    MNIST = "MNIST"
    CELEBA_HQ = "Celeba_HQ"
    AFHQ = "AFHQ"

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default="Celeba_HQ",
                        help='name of the dataset')
    parser.add_argument('--image_size', type=int, default=128,
                        help='image size')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='number of channels of the input')

    # Training
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--commit_w', type=float, default=1.0,
                        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')
    parser.add_argument('--mse_w', type=float, default=0.9,
                        help='weight for the mse reconstruction loss')
    parser.add_argument('--percept_w', type=float, default=0.1,
                        help='weight for the perceptual loss')
    parser.add_argument('--vgg_layers', type=str, default="8_22",
                        help='Target VGG16 layers for perceptual loss')
    parser.add_argument('--save_every', type=int, default=1,
                        help='interval for saving input/reconstruction results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='set the device (cpu or cuda)')

    # P_Hologen hyperparams
    parser.add_argument('--z_dim', type=int, default=256,
                        help='number of channels for the latent vector')
    parser.add_argument('--k', type=int, default=512,
                        help='number of latent vectors in the codebook')

    # Hologram
    parser.add_argument('--scale_factor', type=float, default=0.95,
                        help='scale factor for intensity extraction')
    parser.add_argument('--prop_dist', type=str, default='21.5_mm',
                        help='propagation distance for ASM')
    parser.add_argument('--feature_size', type=float, default=6.4,
                        help='pixel pitch of the SLM')

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise AssertionError("GPU Not available")

    train_p_hologen(args)
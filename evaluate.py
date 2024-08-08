import math
import torch
import torchvision
from torch.utils.data import DataLoader

from modules.p_hologen import P_Hologen
from modules.pixelsnail import PixelSNAIL
from pathlib import Path

import argparse
from tqdm import tqdm

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance

import heapq
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def process_tensor(tensor):
    processed_tensor = tensor.clamp(0., 1.)
    return processed_tensor

def sample_latents(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)# [num samples, 16, 16]
    cache = {}
    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row

def evaluate(opts):
    dataset_name = opts.dataset_name
    input_channels = opts.in_channels
    img_size = opts.image_size
    batch_size = opts.batch_size
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
    scale_factor = opts.scale_factor

    mse_w = opts.mse_w
    percept_w = opts.percept_w

    setting_memo = "samples_{}_{}ch_z_dim{}_recon_w_m{}_p{}_chmult_{}".format(
        dataset_name,
        input_channels,
        z_dim,
        mse_w,
        percept_w,
        txt_for_save
    )

    print("setting_memo: {}".format(setting_memo))

    result_output_path = "training_results/samples/p_hologen/{}/{}".format(img_size, setting_memo)
    Path(result_output_path).mkdir(parents=True, exist_ok=True)

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

    elif dataset_name == AFHQ:

        trainset = torchvision.datasets.ImageFolder('datasets/afhq/train', transform=img_transform)

        valset = torchvision.datasets.ImageFolder('datasets/afhq/val', transform=img_transform)

    else:
        raise AssertionError(
            "Invalid dataset name: {}, available: MNIST, Celeba_HQ, AFHQ".format(dataset_name))

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    p_hologen = P_Hologen(
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

    p_hologen_ckpt = "ckpt_outputs/p_hologen/{}/p_hologen_{}_{}ch_z_dim256_recon_w_m{}_p{}_chmult_{}_prop_{}_pixel_{}/best_model.pth".format(
        img_size,
        dataset_name,
        input_channels,
        mse_w,
        percept_w,
        txt_for_save,
        opts.prop_dist,
        opts.feature_size
    )

    p_hologen.load_state_dict(torch.load(p_hologen_ckpt))
    print("Loaded P_Hologen from {}".format(p_hologen_ckpt))

    p_hologen = p_hologen.to(device)

    p_hologen.eval()

    p_hologen_params = sum(p.numel() for p in p_hologen.parameters())
    print(f"Number of P-Hologen parameters: {p_hologen_params}")

    if opts.gen_samples or opts.eval_samples:
        pixsnail_ckpt = "ckpt_outputs/PixSnail/{}/pixsnail_{}_{}ch_z_dim{}_recon_w_m{}_p{}_chmult_{}_prop_{}_pixel_{}/best_model.pth".format(
            img_size,
            dataset_name,
            input_channels,
            z_dim,
            mse_w,
            percept_w,
            txt_for_save,
            opts.prop_dist,
            opts.feature_size
        )

        latent_size = [16, 16]

        pixel_snail = PixelSNAIL(
            shape=latent_size,
            n_class=k,
            channel=z_dim,
            kernel_size=5,
            n_block=4,
            n_res_block=4,
            res_channel=256,
            attention=True
        )

        pixel_snail.load_state_dict(torch.load(pixsnail_ckpt))
        print("Loaded PixelSnail from {}".format(pixsnail_ckpt))

        pixel_snail = pixel_snail.to(device)

        pixel_snail.eval()

    ssim = StructuralSimilarityIndexMeasure(data_range=(0., 1.)).to(device)

    psnr = PeakSignalNoiseRatio(data_range=(0., 1.)).to(device)

    if opts.gen_samples:

        num_samples = opts.num_samples

        sampled_latents = sample_latents(pixel_snail, device, num_samples, latent_size, opts.temp)

        sample_pohs, recon_samples = p_hologen._decode(sampled_latents)

        for b_idx in range(num_samples):
            img = recon_samples[b_idx]
            torchvision.utils.save_image(img, result_output_path + "/sample_img{}.png".format(b_idx))
            for c_idx in range(input_channels):
                poh = sample_pohs[b_idx][c_idx]
                normalized_poh = phase_transform((poh.clone() + math.pi) / (2 * math.pi))
                normalized_poh.save(result_output_path + "/sample_poh{}_{}ch.png".format(b_idx, c_idx))

        print("Finished generating samples")

    if opts.eval_recons:
        total_psnr = 0
        total_ssim = 0

        cnt = 0
        for img, label in tqdm(val_dataloader):

            B, C, H, W = img.shape

            if input_channels == 1 and C == 3:
                img = img[:, 1:2, :, :]

            img = img.to(device)

            recon_poh, recon_img, _, _ = p_hologen(img)

            processed_pred = process_tensor(recon_img)
            processed_gt = process_tensor(img)

            psnr_score = psnr(processed_pred, processed_gt)

            ssim_score = ssim(processed_pred, processed_gt)

            total_psnr += psnr_score

            total_ssim += ssim_score

            cnt += 1

        avg_psnr = total_psnr / cnt

        avg_ssim = total_ssim / cnt

        print("avg PSNR: {}\navg SSIM: {}".format(avg_psnr, avg_ssim))

    if opts.make_recons:

        for img, label in val_dataloader:

            B, C, H, W = img.shape

            if input_channels == 1 and C == 3:
                img = img[:, 1:2, :, :]

            for b_idx in range(batch_size):
                cur_img = img[b_idx]
                torchvision.utils.save_image(cur_img, result_output_path + "/val_input_img{}.png".format(b_idx))

            img = img.to(device)

            recon_poh, recon_img, _, _ = p_hologen(img)

            for b_idx in range(batch_size):
                recon_ = recon_img[b_idx]
                torchvision.utils.save_image(recon_, result_output_path + "/val_recon_img{}.png".format(b_idx))
                for c_idx in range(input_channels):
                    poh = recon_poh[b_idx][c_idx]
                    normalized_poh = phase_transform((poh.clone() + math.pi) / (2 * math.pi))
                    normalized_poh.save(result_output_path + "/val_recon_poh{}_{}ch.png".format(b_idx, c_idx))

            break
        print("Reconstructed a batch of validation set")

    if opts.eval_samples:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        for img, label in val_dataloader:

            B, C, H, W = img.shape

            if input_channels == 1 and C == 3:
                img = img[:, 1:2, :, :]

            B, C, H, W = img.shape

            img = img.to(device)

            if C == 1:
                gt_img = img.expand(B, 3, H, W)
            else:
                gt_img = img

            fid.update(gt_img, real=True)

            recon_poh, recon_img, _, _ = p_hologen(img)

            sampled_latents = sample_latents(pixel_snail, device, batch_size, latent_size, opts.temp)

            sample_pohs, recon_samples = p_hologen._decode(sampled_latents)

            recon_samples = process_tensor(recon_samples)

            B, C, H, W = recon_samples.shape

            if C == 1:
                recon_samples = recon_samples.expand(B, 3, H, W)

            fid.update(recon_samples, real=False)

            break

        fid_score = fid.compute()

        print("FID: {}".format(fid_score))

    if opts.find_nearest:
        image_path = opts.find_nearest
        sample_img = Image.open(image_path)

        sample_img = img_transform(sample_img)

        sample_img = sample_img.unsqueeze(0)

        lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True)

        lpips_top_images = []

        num_neighbors = 6
        for imgs, label in tqdm(train_dataloader):
            for img in imgs:
                # Ensure the training image is on the same device as the model
                img = img.unsqueeze(0)  # Add batch dimension

                # Compute LPIPS score
                lpips_score = lpips(sample_img, img).item()

                # Using a min heap to keep track of top scores
                # The heap stores tuples of (-score, img) since heapq is a min heap
                if len(lpips_top_images) < num_neighbors:
                    heapq.heappush(lpips_top_images, (-lpips_score, img))
                else:
                    heapq.heappushpop(lpips_top_images, (-lpips_score, img))

        for i, (lpips_score, top_img) in enumerate(lpips_top_images):
            torchvision.utils.save_image(top_img, result_output_path + f"/lpips_top_image_{num_neighbors - i}_{lpips_score * -1}.png")

        print("Finished finding the nearest neighbors of {}".format(opts.find_nearest))

    return


if __name__ == '__main__':

    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

    metric_dict = {
        "cm": 1e-2,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
    }

    wavelength_list = (638 * nm, 520 * nm, 450 * nm)

    MNIST = "MNIST"
    FASHION_MNIST = "FashionMNIST"
    CELEBA_HQ = "Celeba_HQ"
    AFHQ = "AFHQ"
    IMAGENET = "Imagenet"

    parser = argparse.ArgumentParser()
    parser.add_argument('--find_nearest', type=str, default="",
                        help='path of the target file to find the nearest neighbors')
    parser.add_argument('--make_recons', action='store_true',
                        help='reconstruct a batch of validation set')
    parser.add_argument('--gen_samples', action='store_true',
                        help='generate samples')
    parser.add_argument('--eval_recons', action='store_true',
                        help='evaluate reconstruction ability with psnr and ssim')
    parser.add_argument('--eval_samples', action='store_true',
                        help='evaluate generation ability with fid')

    parser.add_argument('--dataset_name', type=str, default="Celeba_HQ",
                        help='name of the dataset')
    parser.add_argument('--image_size', type=int, default=128,
                        help='image size')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='number of channels')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--temp', type=float, default=1.0)

    # Training
    parser.add_argument('--mse_w', type=float, default=0.9,
                        help='weight for the mse reconstruction loss')
    parser.add_argument('--percept_w', type=float, default=0.1,
                        help='weight for the perceptual loss')
    parser.add_argument('--device', type=str, default='cuda',
                        help='set the device (cpu or cuda)')

    # P-Hologen hyperparams
    parser.add_argument('--z_dim', type=int, default=256,
                        help='number of channels for the latent vector')
    parser.add_argument('--k', type=int, default=512,
                        help='number of latent vectors in the codebook')

    # Hologram
    parser.add_argument('--scale_factor', type=float, default=0.95,
                        help='scale factor for intensity extraction')
    parser.add_argument('--prop_dist', type=str, default='21.5_mm', # 64: 15.0mm / 128: 21.5 mm / 256: 25.0 mm / 1024: 122.3 mm
                        help='propagation distance for ASM')
    parser.add_argument('--feature_size', type=float, default=6.4,
                        help='pixel pitch of the SLM')

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise AssertionError("GPU Not available")

    with torch.no_grad():
        evaluate(args)

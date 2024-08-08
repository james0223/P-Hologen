import torch
import torchvision
from tqdm import tqdm
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from modules.p_hologen import P_Hologen
from modules.pixelsnail import PixelSNAIL
import sys


def train_pixsnail(opts):
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
    scale_factor = opts.scale_factor

    mse_w = opts.mse_w
    percept_w = opts.percept_w
    max_patience = 15

    setting_memo = "pixsnail_{}_{}ch_z_dim{}_recon_w_m{}_p{}_chmult_{}_prop_{}_pixel_{}".format(
        dataset_name,
        input_channels,
        z_dim,
        mse_w,
        percept_w,
        txt_for_save,
        opts.prop_dist,
        opts.feature_size
    )

    print("Setting memo: {}".format(setting_memo))

    ckpt_output_path = "ckpt_outputs/PixSnail/{}/{}".format(img_size, setting_memo)

    Path(ckpt_output_path).mkdir(parents=True, exist_ok=True)

    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ])

    if dataset_name == MNIST:

        trainset = torchvision.datasets.MNIST(root='datasets',
                                              train=True,
                                              transform=img_transform,
                                              download=True)

    elif dataset_name == CELEBA_HQ:

        trainset = torchvision.datasets.ImageFolder('datasets/celeba_hq/train', transform=img_transform)

    elif dataset_name == AFHQ:

        trainset = torchvision.datasets.ImageFolder('datasets/afhq/train', transform=img_transform)

    else:
        raise AssertionError(
            "Invalid dataset name: {}, available: MNIST, Celeba_HQ, AFHQ".format(dataset_name))

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

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

    p_hologen_ckpt = "ckpt_outputs/p_hologen/{}/p_hologen_{}_{}ch_z_dim{}_recon_w_m{}_p{}_chmult_{}_prop_{}_pixel_{}/best_model.pth".format(
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

    p_hologen.load_state_dict(torch.load(p_hologen_ckpt))
    print("Loaded P_Hologen from {}".format(p_hologen_ckpt))

    p_hologen.eval()

    model = PixelSNAIL(
        shape=(16, 16),
        n_class=k,
        channel=z_dim,
        kernel_size=5,
        n_block=4,
        n_res_block=4,
        res_channel=256,
        attention=True
    ).to(device)

    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = sys.maxsize

    patience = 0

    for epoch in tqdm(range(num_epochs)):
        training_loss = 0
        train_cnt = 0
        for input_imgs, labels in train_dataloader:

            optimizer.zero_grad()

            with torch.no_grad():

                B, C, H, W = input_imgs.shape

                if input_channels == 1 and C == 3:
                    input_imgs = input_imgs[:, 1:2, :, :]

                input_imgs = input_imgs.to(device)

                gt_indices = p_hologen._encode(input_imgs)

            outs, cache = model(gt_indices)

            loss = criterion(outs, gt_indices)

            training_loss += loss

            loss.backward()

            optimizer.step()
            train_cnt += 1

        training_avg_loss = training_loss / train_cnt

        if training_avg_loss < best_loss:
            best_loss = training_avg_loss
            patience = 0
            torch.save(model.state_dict(), "{}/best_model.pth".format(ckpt_output_path))
            print("Average training loss: {}".format(training_avg_loss))
            print("Saved best model for epoch {}".format(epoch))
            print("Patience reset to 0")
        else:
            patience += 1
            print("No improvement for epoch {}, patience: {}".format(epoch, patience))

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
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate for pixelsnail')
    parser.add_argument('--mse_w', type=float, default=0.9,
                        help='weight for the mse reconstruction loss')
    parser.add_argument('--percept_w', type=float, default=0.1,
                        help='weight for the perceptual loss')
    parser.add_argument('--device', type=str, default='cuda',
                        help='set the device (cpu or cuda)')

    # p_hologen hyperparams
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

    train_pixsnail(args)
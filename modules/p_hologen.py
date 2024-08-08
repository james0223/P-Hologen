import torch
import torch.nn as nn
import math
from modules.propagation_ASM import propagation_ASM
import utils

from modules.functions import vq, vq_st
from modules.layers import Encoder, Decoder


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class P_Hologen(nn.Module):
    def __init__(self,
                 # VQ
                 input_channels=1,
                 z_dim=256,
                 K=512,
                 ch_mult=[1, 2, 4],
                 attn_resolutions=[16],

                 # Hologram
                 img_size=64,
                 feature_size=None,
                 wavelength_list=None,
                 distance_list=None,
                 scale_output=None,
                 pad_type="zero",

                 # etc
                 device="cpu"
                 ):
        super(P_Hologen, self).__init__()

        if input_channels not in (1, 3):
            raise AssertionError("Only the input channels of either 1 or 3 is allowed.")

        self.img_size = img_size
        self.num_channels = input_channels
        self.device = device

        self.pad_type = pad_type
        self.feature_size = feature_size
        self.wavelength_list = wavelength_list
        self.distance_list = distance_list
        self.scale_output = scale_output

        self.input_resolution = [self.img_size, self.img_size]

        self.conv_size = [i * 2 for i in self.input_resolution]

        self.encoder = Encoder(
            ch=128,
            out_ch=1,
            ch_mult=ch_mult,
            num_res_blocks=2,
            attn_resolutions=attn_resolutions,
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=input_channels,
            resolution=img_size,
            z_channels=z_dim,
            double_z=False
        )
        self.decoder = Decoder(
            ch=128,
            out_ch=input_channels,
            ch_mult=ch_mult,
            num_res_blocks=2,
            attn_resolutions=attn_resolutions,
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=1,
            resolution=img_size,
            z_channels=z_dim,
            give_pre_end=False
        )

        self.codebook = VQEmbedding(K, z_dim)

        self.apply(weights_init)

        self.hard_tan = nn.Hardtanh(-math.pi, math.pi)

        self.prop = propagation_ASM

        self.H = [None, None, None]

    def recon_hologram(self, poh, idx):

        ones_amp = torch.ones_like(poh)

        f_real, f_imag = utils.polar_to_rect(ones_amp, poh)

        poh_complex = torch.complex(f_real, f_imag)

        if self.H[idx] is None:
            self.H[idx] = self.prop(poh_complex,
                                    self.feature_size,
                                    self.wavelength_list[idx],
                                    self.distance_list[idx],
                                    return_H=True,
                                    linear_conv=True)
            self.H[idx] = self.H[idx].to(self.device).detach()
            self.H[idx].requires_grad = False

        if self.pad_type == 'zero':
            pad_val = 0
        else:  # median
            pad_val = torch.median(torch.pow((poh_complex ** 2).sum(-1), 0.5))

        padded_poh_complex = utils.pad_image(poh_complex, self.conv_size, padval=pad_val, stacked_complex=False)

        # Image Recon
        back_prop = torch.fft.fftn(utils.ifftshift(padded_poh_complex), dim=(-2, -1), norm='ortho')

        back_prop = self.H[idx] * back_prop

        back_prop = utils.fftshift(torch.fft.ifftn(back_prop, dim=(-2, -1), norm='ortho'))

        cropped_back_prop = utils.crop_image(back_prop, self.input_resolution, pytorch=True, stacked_complex=False)

        # output_lin_intensity = torch.sum(cropped_back_prop.abs() ** 2 * self.scale_output, dim=1, keepdim=True)
        output_lin_intensity = cropped_back_prop.abs() ** 2 * self.scale_output

        recon_img = torch.pow(output_lin_intensity, 0.5)

        return recon_img

    def _encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def _decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)

        poh = self.decoder(z_q_x)

        poh = self.hard_tan(poh)

        if self.num_channels == 1:
            # grayscale
            recon_img = self.recon_hologram(poh, 1)
        else:
            # 3-channel RGB
            recon_img = []
            for idx in range(3):
                target_poh = poh[:, idx, :, :].unsqueeze(1)
                recon = self.recon_hologram(target_poh, idx)
                recon_img.append(recon)
            recon_img = torch.cat(recon_img, dim=1)

        return poh, recon_img

    def forward(self, x):
        z_e_x = self.encoder(x)

        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)

        poh = self.decoder(z_q_x_st)

        poh = self.hard_tan(poh)

        if self.num_channels == 1:
            # grayscale
            recon_img = self.recon_hologram(poh, 1)
        else:
            # 3-channel RGB
            recon_img = []
            for idx in range(3):
                target_poh = poh[:, idx, :, :].unsqueeze(1)
                recon = self.recon_hologram(target_poh, idx)
                recon_img.append(recon)
            recon_img = torch.cat(recon_img, dim=1)

        return poh, recon_img, z_e_x, z_q_x
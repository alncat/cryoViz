'''Pytorch models'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from . import sync_batchnorm
import matplotlib.pyplot as plt
from . import pose_encoder
from . import decoders
from . import fft
from . import lie_tools
from . import utils
from . import lattice
from . import mrc
from . import symm_groups
from . import unet
from . import healpix_sampler

log = utils.log
ALIGN_CORNERS = utils.ALIGN_CORNERS

class HetOnlyVAE(nn.Module):
    # No pose inference
    def __init__(self, lattice, # Lattice object
            qlayers, qdim,
            players, pdim,
            in_dim, zdim = 1,
            encode_mode = 'resid',
            enc_mask = None,
            enc_type = 'linear_lowf',
            enc_dim = None,
            domain = 'fourier',
            activation = nn.ReLU,
            ref_vol = None,
            Apix = 1.,
            ctf_grid = None,
            template_type = None,
            warp_type = None,
            num_struct = 1,
            deform_emb_size = 2,
            device = None,
            symm = None,
            render_size=140,
            downfrac=0.5,
            down_vol_size=None):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        self.encode_mode = encode_mode
        self.num_struct = num_struct
        self.fixed_deform = False
        self.device = device
        self.render_size = int((lattice.D - 1)*downfrac)//2*2
        self.down_vol_size = int(self.render_size*0.85)//2*2

        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, zdim*2)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(in_dim,
                            qlayers, # nlayers
                            qdim,  # hidden_dim
                            zdim*2, # out_dim
                            activation)
        elif encode_mode == 'mlp':
            self.encoder = MLP(in_dim,
                            qlayers,
                            qdim, # hidden_dim
                            zdim*2, # out_dim
                            activation) #in_dim -> hidden_dim
        elif encode_mode == 'tilt':
            self.encoder = TiltEncoder(in_dim,
                            qlayers,
                            qdim,
                            zdim*2,
                            activation)
        elif encode_mode == 'fixed':
            #self.zdim = 256
            self.encoder = FixedEncoder(self.num_struct, self.zdim)
            self.pose_encoder = pose_encoder.PoseEncoder(image_size=128)
        elif encode_mode == 'fixed_blur':
            #self.zdim = 256
            self.encoder = FixedEncoder(self.num_struct, self.zdim)
            #self.affine_encoder = Encoder(1, self.num_struct - 1, lattice.D)
        elif encode_mode == 'deform':
            #self.zdim = 256
            self.encoder = FixedEncoder(self.num_struct, self.zdim)
            self.fixed_deform = True
        elif encode_mode == 'grad':
            self.encoder = Encoder(self.zdim, lattice.D, crop_vol_size=110)
            self.batch_norm = sync_batchnorm.SynchronizedBatchNorm1d(self.zdim, eps=1e-5, affine=False)
            #self.shape_encoder = pose_encoder.PoseEncoder(image_size=128, mode="shape")
            self.fixed_deform = True
        else:
            raise RuntimeError('Encoder mode {} not recognized'.format(encode_mode))
        self.warp_type = warp_type

        self.encode_mode = encode_mode
        self.vanilla_dec = enc_type == "vanilla"
        self.template_type = template_type
        self.symm = symm
        self.deform_emb_size = deform_emb_size
        self.decoder = get_decoder(3+zdim, lattice.D, players, pdim, domain, enc_type, enc_dim,
                                   activation, ref_vol=ref_vol, Apix=Apix,
                                   template_type=self.template_type, warp_type=self.warp_type,
                                   symm=self.symm, ctf_grid=ctf_grid,
                                   fixed_deform=self.fixed_deform, deform_emb_size=self.deform_emb_size,
                                   render_size=self.render_size, down_vol_size=self.down_vol_size)

    @classmethod
    def load(self, config, weights=None, device=None):
        '''Instantiate a model from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            HetOnlyVAE instance, Lattice instance
        '''
        cfg = utils.load_pkl(config) if type(config) is str else config
        c = cfg['lattice_args']
        lat = lattice.Lattice(c['D'], extent=c['extent'])
        c = cfg['model_args']
        if c['enc_mask'] > 0:
            enc_mask = lat.get_circular_mask(c['enc_mask'])
            in_dim = int(enc_mask.sum())
        else:
            assert c['enc_mask'] == -1
            enc_mask = None
            in_dim = lat.D**2
        activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c['activation']]
        model = HetOnlyVAE(lat,
                          c['qlayers'], c['qdim'],
                          c['players'], c['pdim'],
                          in_dim, c['zdim'],
                          encode_mode=c['encode_mode'],
                          enc_mask=enc_mask,
                          enc_type=c['pe_type'],
                          enc_dim=c['pe_dim'],
                          domain=c['domain'],
                          activation=activation)
        if weights is not None:
            ckpt = torch.load(weights)
            model.load_state_dict(ckpt['model_state_dict'])
        if device is not None:
            model.to(device)
        return model, lat

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def encode(self, *img):
        img = (x.view(x.shape[0],-1) for x in img)
        if self.enc_mask is not None:
            img = (x[:,self.enc_mask] for x in img)
        z = self.encoder(*img)
        #if self.encode_mode == 'fixed':
        #    z = torch.tile(self.encoder, (x.shape[0], 1))
        return z[:,:self.zdim], z[:,self.zdim:]

    def cat_z(self, coords, z):
        '''
        coords: Bx...x3
        z: Bxzdim
        '''
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1]*(coords.ndimension()-2)), self.zdim)
        z = torch.cat((coords,z.expand(*coords.shape[:-1],self.zdim)),dim=-1)
        return z

    def decode(self, coords, z, mask=None):
        '''
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        '''
        return self.decoder(self.cat_z(coords,z))

    def get_fixedcode(self):
        return self.encoder()

    def vanilla_encode(self, img, rots=None, trans=None):
        if self.encode_mode == 'fixed':
            z = self.encoder()
            encout = {'encoding': None}
        elif self.encode_mode == 'fixed_blur':
            #split encodings to template and blur kernel
            zs = self.encoder()
            z = zs[:1, :]
            #print(img.shape)
            encout = {"encoding": zs[1:, :]}
            #print(z.shape, encout['encoding'].shape)
        elif self.encode_mode == "grad":
            encout = self.encoder(img, rots, trans, losslist=["kldiv"])
            mu     = encout["z_mu"]
            logstd = encout["z_logstd"]
            z  = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
            encout["encoding"] = z
        return z, encout

    def vanilla_decode(self, rots, trans, z=None, save_mrc=False, eulers=None,
                       ref_fft=None, ctf=None, encout=None, others=None, mask=None):
        in_template = None
        if self.encode_mode != 'deform':
            #randomly perturb rotation
            #euler_rot = lie_tools.euler_to_SO3(eulers)
            #print((euler_rot - rots).abs().max())
            #rots_perturb = lie_tools.random_biased_SO3(rots.shape[0], bias=3., device=rots.get_device())
            #print(rots_perturb)
            #rots_perturb = rots #@ rots_perturb
            #print(encout['encoding'].shape, self.encoder().shape)
            #z = encout['encoding']
            #mu = self.batch_norm(encout['z_mu'])
            #encout["z_mu"]     = mu
            #z, encout = self.vanilla_encode(img, rots, trans)
            pass
        else:
            #for deform embdding, the encoding will come from z
            encout = {'encoding': None}
        decout = self.decoder(rots, trans, z=z, in_template=in_template, save_mrc=save_mrc,
                              euler=eulers, ref_fft=ref_fft, ctf=ctf, others=others)
        decout["y_recon_ori"] = decout["y_recon"]
        y_recon_ori = decout["y_recon"]*utils.crop_image(mask, self.down_vol_size)
        pad_size = (self.render_size - self.down_vol_size)//2
        y_recon_ori = F.pad(y_recon_ori, (pad_size, pad_size, pad_size, pad_size))

        if "ctf" in others:
            #print("ctf: ", ctf[0], others["ctf"].shape)
            ctf = torch.cat([ctf.unsqueeze(1), others["ctf"]], dim=1)
            #ref_fft = torch.cat([ref_fft, others["y_fft"]], dim=1)
        else:
            ctf = ctf.unsqueeze(1)
        #print(y_recon_ori.shape, ctf.shape)
        #y_recon_ori  = torch.view_as_complex(y_recon_fft)
        #decout["y_recon_fft"] = y_recon_ori*ctf # ctf is (B, C, H, W) (B, 1, H, W, 2) x (B, 1, H, W, 1)
        #y_recon_fft   = decout["y_recon_fft"]
        #y_ref_fft   = torch.view_as_complex(decout["y_ref_fft"])
        #print(y_recon_fft.shape, y_ref_fft.shape, ctf.shape)
        # convert to image
        #y_recon_fft_s = torch.fft.fftshift(y_recon_fft, dim=(-2))
        #y_recon = fft.torch_ifft2_center(y_recon_fft_s)

        # put zero frequency on border
        #y_recon_fft_s = torch.fft.fftshift(y_recon_ori, dim=(-2))
        #y_recon_ori = fft.torch_ifft2_center(y_recon_fft_s)*mask
        y_recon_fft = fft.torch_fft2_center(y_recon_ori[:, :ctf.shape[1], ...])*ctf
        # put zero frequency in center
        #y_recon_fft = torch.fft.fftshift(y_recon_fft, dim=(-2))*ctf
        # put zero frequency on border
        #y_recon_fft = torch.fft.fftshift(y_recon_fft, dim=(-2))

        #y_ref_fft_s = torch.fft.fftshift(ref_fft, dim=(-2))
        #y_ref = fft.torch_ifft2_center(ref_fft)
        #print(y_ref.shape)

        decout["y_recon_fft"] = torch.view_as_real(y_recon_fft)
        #decout["y_ref_fft"] = torch.view_as_real(ref_fft)

        decout["y_recon"] = fft.torch_ifft2_center(y_recon_fft)
        #decout["y_recon_ori"] = y_recon_ori
        #decout["y_ref"] = y_ref

        if self.encode_mode in ["fixed"]:
            #decout["probs"] = torch.ones(z.shape[0], 1, 1).to(z.get_device())
            B = z.shape[0]
            latent_dist = -(z.unsqueeze(1) - z.unsqueeze(0)).pow(2).sum(-1)*0.5
            #remove diagonal
            diag_mask = ~torch.eye(B, dtype=bool).to(z.get_device())
            latent_dist = latent_dist.masked_select(diag_mask).view(B, B-1)
            print(F.softmax(latent_dist, dim=-1))
            #get probs
            latent_log_probs = F.log_softmax(latent_dist, dim=-1)
            #decout["probs"] = dist.exp().detach().unsqueeze(-1)
            #print(decout["probs"])
            #stack and pass to encoder
            B, C, H, W = y_recon.shape #(B is batch, C represent different views)
            #compute transition probabilities
            dist = -y_ref.pow(2)*0.5 + y_recon.detach()*y_ref - y_recon.detach().pow(2)*0.5
            dist = dist.sum(dim=(-1, -2))*32/128**2
            #remove diagonal
            #dist_diag = torch.diagonal(dist)
            dist = dist.masked_select(diag_mask).view(B, B-1)
            probs = F.softmax(dist, dim=-1)
            #print(probs)
            decout["losses"]["pairkld"] = -probs*latent_log_probs
            #keep diagonal
            decout["y_recon_fft"] = torch.view_as_real(torch.diagonal(y_recon_fft).permute(dims=[2,0,1]).unsqueeze(1))
            decout["y_ref_fft"]   = torch.view_as_real(torch.diagonal(y_ref_fft).permute(dims=[2,0,1]).unsqueeze(1))

            decout["y_recon"] = torch.diagonal(y_recon).permute(dims=[2,0,1]).unsqueeze(1)
            decout["y_ref"]   = torch.diagonal(y_ref).permute(dims=[2,0,1]).unsqueeze(1)
            #print(decout["y_recon"].shape)
            #y_recon = y_recon.view(B*C, 1, H, W)
            #y_recon = y_recon + torch.randn(y_recon.shape).to(y_recon.get_device())
            #y_ref   = y_ref.view(B*C, 1, H, W)
            #print(y_recon.shape, y_ref.shape, ctf.shape)
            #ys_detached = torch.cat([y_recon.detach(), y_ref], dim=1)
            #print(ys.shape)
            #fake_probs = self.shape_encoder(ys_detached).view(B, C, 1)
            #real_probs = self.shape_encoder(y_ref)
            #rots = rots.unsqueeze(0).repeat(B, 1, 1).view(B*C, -1)
            #encout = self.encoder(y_recon.detach(), rots, None, losslist=[])
            #decout["kldfake"] = fake_probs
            #construct discriminator loss
            #decout["losses"]["lossF"] = 0.5*fake_probs[:, 0]**2
            #decout["losses"]["lossR"] = 0.5*fake_probs[:, 1:]**2
            #generator loss
            #ys = torch.cat([y_recon, y_ref], dim=1)
            #probs = self.shape_encoder(ys).view(B, C, 1)
            #decout["ys"] = ys
            #print(probs)
            #trans = self.pose_encoder(ys)
            #print(trans.shape)
            #ps = self.decoder.fourier_transformer.translate_ft(trans).squeeze(0).unsqueeze(1)
            #ps = utils.crop_fft(ps, 128)
            #print(trans)

        return decout

    # Need forward func for DataParallel -- TODO: refactor
    def forward(self, *args, **kwargs):
        if self.vanilla_dec:
            return self.vanilla_decode(*args, **kwargs)
        else:
            return self.decode(*args, **kwargs)

    def save_mrc(self, filename, enc=None, Apix=1.):
        if self.vanilla_dec:
            if enc is not None:
                self.decoder.save(filename, z=enc, Apix=Apix)

    def get_images(self, rots, trans):
        assert self.vanilla_dec
        return self.decoder.get_images(self.encoder(), rots, trans)

    def get_vol(self, z):
        if self.vanilla_dec:
            encoding = None
            if self.encode_mode == 'fixed':
                z = self.encoder()
            elif self.encode_mode == 'fixed_blur':
                z = self.encoder()
                #encout = self.affine_encoder(img)
                #encoding = encout['encoding']
                #z += encoding
            return self.decoder.get_vol(z=z)

def load_decoder(config, weights=None, device=None):
    '''
    Instantiate a decoder model from a config.pkl

    Inputs:
        config (str, dict): Path to config.pkl or loaded config.pkl
        weights (str): Path to weights.pkl
        device: torch.device object

    Returns a decoder model
    '''
    cfg = utils.load_pkl(config) if type(config) is str else config
    c = cfg['model_args']
    D = cfg['lattice_args']['D']
    activation={"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c['activation']]
    model = get_decoder(3, D, c['layers'], c['dim'], c['domain'], c['pe_type'], c['pe_dim'], activation)
    if weights is not None:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt['model_state_dict'])
    if device is not None:
        model.to(device)
    return model

def get_decoder(in_dim, D, layers, dim, domain, enc_type, enc_dim=None, activation=nn.ReLU, templateres=128,
                ref_vol=None, Apix=1., template_type=None, warp_type=None,
                symm=None, ctf_grid=None, fixed_deform=False, deform_emb_size=2, render_size=140, down_vol_size=140):
    if enc_type == 'none':
        if domain == 'hartley':
            model = ResidLinearMLP(in_dim, layers, dim, 1, activation)
            ResidLinearMLP.eval_volume = PositionalDecoder.eval_volume # EW FIXME
        else:
            model = FTSliceDecoder(in_dim, D, layers, dim, activation)
        return model
    elif enc_type == 'vanilla':
        #model = VanillaDecoder
        #if template_type is None:
        #    assert ref_vol is not None
        return VanillaDecoder(D, ref_vol, Apix, template_type=template_type, warp_type=warp_type,
                              symm_group=symm, ctf_grid=ctf_grid,
                              fixed_deform=fixed_deform,
                              deform_emb_size=deform_emb_size,
                              zdim=in_dim - 3, render_size=render_size, down_vol_size=down_vol_size)
    else:
        model = PositionalDecoder if domain == 'hartley' else FTPositionalDecoder
        return model(in_dim, D, layers, dim, activation, enc_type=enc_type, enc_dim=enc_dim)

class FixedEncoder(nn.Module):
    def __init__(self, num_struct=1, in_dim=256):
        super(FixedEncoder, self).__init__()
        self.in_dim = in_dim
        self.num_struct=num_struct
        self.register_buffer('encoding1', torch.randn((self.num_struct, self.in_dim)))

    def forward(self,):
        return self.encoding1

class ConvTemplate(nn.Module):
    def __init__(self, in_dim=256, outchannels=1, templateres=128):

        super(ConvTemplate, self).__init__()

        self.zdim = in_dim
        self.outchannels = outchannels
        self.templateres = templateres

        self.template1 = nn.Sequential(nn.Linear(self.zdim, 512), nn.LeakyReLU(0.2),
                                       nn.Linear(512, 2048), nn.LeakyReLU(0.2))
        template2 = []
        inchannels, outchannels = 2048, 1024
        template2.append(nn.ConvTranspose3d(inchannels, outchannels, 2, 2, 0))
        template2.append(nn.LeakyReLU(0.2))

        inchannels, outchannels = 1024, 512
        template2.append(nn.ConvTranspose3d(inchannels, outchannels, 2, 2, 0))
        template2.append(nn.LeakyReLU(0.2))
        self.template2 = nn.Sequential(*template2)

        inchannels, outchannels = 512, 256
        template3 = []
        template4 = []
        for i in range(int(np.log2(self.templateres)) - 3): #3):
            if i < 3:
                template3.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
                template3.append(nn.LeakyReLU(0.2))
            else:
                template4.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
                template4.append(nn.LeakyReLU(0.2))
            inchannels = outchannels
            outchannels = inchannels//2 #max(inchannels // 2, 16)
        self.template3 = nn.Sequential(*template3)
        self.template4 = nn.Sequential(*template4)
        #self.conv_out = nn.Conv3d(inchannels, 1, 3, 1, 1)
        self.conv_out = nn.ConvTranspose3d(inchannels, 1, 4, 2, 1)
        #self.conv_out = nn.ConvTranspose3d(inchannels, 1, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        for m in [self.template1, self.template2, self.template3, self.template4]:
            utils.initseq(m)
        utils.initmod(self.conv_out, gain=1./np.sqrt(templateres))

    def forward(self, encoding):
        template2 = self.template2(self.template1(encoding).view(-1, 2048, 1, 1, 1))
        template3 = self.template3(template2)
        template3 = F.interpolate(template3, size=24, mode="trilinear", align_corners=ALIGN_CORNERS)
        template = self.template4(template3)
        out =  self.conv_out(template)
        return out #self.conv_out(template)

class AffineMixWeight(nn.Module):
    def __init__(self, in_dim=8, out_dim=3, out_size=32):
        super(AffineMixWeight, self).__init__()

        self.quat = utils.Quaternion()
        self.out_dim = out_dim

        inchannels = 8
        self.inchannels = inchannels
        self.warpf = nn.Sequential(
                nn.Linear(in_dim, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, inchannels*2*2*2), nn.LeakyReLU(0.2)
                )
        outchannels = self.out_dim
        upsample = []
        n_layers = int(np.log2(out_size) - 1)
        for i in range(n_layers - 1):
            upsample.append(nn.ConvTranspose3d(inchannels, inchannels, 4, 2, 1))
            upsample.append(nn.LeakyReLU(0.2))
        upsample.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
        self.upsample = nn.Sequential(*upsample)

        utils.initseq(self.warpf)
        utils.initseq(self.upsample)

    def forward(self, encoding):
        init_vol = self.warpf(encoding).view(-1, self.inchannels, 2, 2, 2)
        out = self.upsample(init_vol)
        return out

class Encoder(nn.Module):
    def __init__(self, zdim, D, crop_vol_size):
        super(Encoder, self).__init__()

        self.zdim = zdim
        self.inchannels = 1
        self.vol_size = D - 1
        self.crop_vol_size = crop_vol_size #int(160*self.scale_factor)
        #downsample volume
        self.transformer = SpatialTransformer(self.crop_vol_size)
        self.out_dim = (self.crop_vol_size)//128 + 1

        #self.init_conv = nn.Sequential(
        #                    nn.Conv2d(1, 8, 4, 2, 1),
        #                    nn.LeakyReLU(0.1)
        #                )

        downsample = []
        n_layers = int(np.log2(128//2))
        inchannels = 1
        outchannels = 32
        for i in range(n_layers):
            downsample.append(nn.Conv3d(inchannels, outchannels, 4, 2, 1))
            downsample.append(nn.LeakyReLU(0.2))
            inchannels = outchannels
            #if inchannels == outchannels:
            outchannels = min(inchannels * 2, 512)
            #else:
        self.out_channels = inchannels
        #downsample.append(nn.Conv3d(inchannels, self.out_channels, 4, 2, 1))
        #downsample.append(nn.LeakyReLU(0.2))
        self.down1 = nn.Sequential(*downsample)
        #downsample.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))

        #self.down1 = nn.Sequential(
        #        nn.Conv3d(self.inchannels, 16, 4, 2, 1),    nn.LeakyReLU(0.2),#40
        #        nn.Conv3d(16, 16, 4, 2, 1),   nn.LeakyReLU(0.2),#20
        #        nn.Conv3d(16, 32, 4, 2, 1),  nn.LeakyReLU(0.2),#10
        #        nn.Conv3d(32, 32, 4, 2, 1), nn.LeakyReLU(0.2),#5
        #        nn.Conv3d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),#2
        #        nn.Conv3d(64, self.out_channels, 4, 2, 1), nn.LeakyReLU(0.2))#1
        self.down2 = nn.Sequential(
                nn.Linear(self.out_channels * self.out_dim ** 3, 512), nn.LeakyReLU(0.2))

        self.mu = nn.Linear(512, self.zdim)
        self.logstd = nn.Linear(512, self.zdim)

        self.quat = nn.Linear(512, 3)

        utils.initseq(self.down1)
        utils.initseq(self.down2)
        utils.initmod(self.mu)
        utils.initmod(self.logstd)
        utils.initmod(self.quat)

    def forward(self, x, rots, trans, losslist=[]):
        #2d to 3d suppose x is (N, 1, H, W)
        B = x.shape[0]
        #x = self.init_conv(x).unsqueeze(2)
        x = utils.crop_image(x, self.crop_vol_size).unsqueeze(2)

        x3d = x.repeat(1, 1, self.crop_vol_size, 1, 1) #(N, D, H, W)
        #print(x3d.shape)
        encs = []
        x3d_downs = []
        for i in range(B):
            #rotate the gradient
            pos = self.transformer.rotate(rots[i].T)
            #downsample the gradient
            x3d_down = F.grid_sample(x3d[i:i+1], pos, align_corners=ALIGN_CORNERS)
            x3d_downs.append(x3d_down)
            #pass through convolution nn
        x3d_downs = torch.cat(x3d_downs, dim=0)
        #print(x3d_downs.shape)
        enc1 = self.down1(x3d_downs)
        #print(enc1.shape, self.out_channels, self.out_dim)
        encs = enc1.view(B, self.out_dim ** 3 *self.out_channels)
        encs = self.down2(encs)

        mu = self.mu(encs)
        if self.training and "kldiv" in losslist:
            logstd = self.logstd(encs)
            #z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
            #z = mu
        else:
            logstd = None
            z = mu

        quat = self.quat(encs)
        ones = torch.ones(mu.shape[0], 1).to(mu.get_device())*2
        quat = torch.cat([ones, quat], dim=1)
        rot = lie_tools.quaternions_to_SO3_wiki(quat)

        losses = {}
        if "kldiv" in losslist:
            #losses["kldiv"] = torch.mean(mu**2, dim=-1)
            losses["mu2"] = torch.sum(mu**2, dim=-1)
            losses["std2"] = torch.sum(torch.exp(2*logstd), dim=-1)
            #losses["kldiv"] = torch.mean(- logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1)
            #losses["kldiv"] = torch.sum(-logstd, dim=-1) + 0.5*losses["std2"] + 0.5*losses["mu2"]

        return {"z_mu": mu, "losses": losses, "z_logstd": logstd, "rot_dev": rot}


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, normalize=True, use_fourier=False, mode='bilinear', render_size=180):
        super().__init__()

        self.mode = mode

        # create sampling grid
        #vectors = [torch.arange(0, s) for s in size]

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict

        self.templateres = size
        self.normalize = normalize
        self.use_fourier = use_fourier
        self.render_size = render_size
        if self.normalize:
            zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres), indexing='ij')
        else:
            zgrid, ygrid, xgrid = np.meshgrid(np.arange(self.templateres),
                                  np.arange(self.templateres),
                                  np.arange(self.templateres), indexing='ij')
        #xgrid is the innermost dimension (-1, ..., 1)
        self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))
        x_idx = torch.linspace(-1., 1., self.render_size) #[-s, s)
        grid  = torch.meshgrid(x_idx, x_idx, indexing='ij')
        xgrid = grid[1] #change fast [[0,1,2,3]]
        #ygrid = torch.roll(grid[0], shifts=(self.x_size), dims=(0)) #fft shifted, center at the corner
        ygrid = grid[0]

        zgrid = torch.zeros_like(xgrid)
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("grid2d", grid)

        x_idx = torch.linspace(-1., 1., self.templateres) #[-s, s)
        grid  = torch.meshgrid(x_idx, x_idx, indexing='ij')
        xgrid = grid[1] #change fast [[0,1,2,3]]
        ygrid = grid[0]
        zgrid = torch.zeros_like(xgrid)
        grid = torch.stack([xgrid, ygrid, zgrid], dim=-1).unsqueeze(0).unsqueeze(0)
        #self.register_buffer("gridz", grid)

    def rotate(self, rot):
        return self.grid @ rot #(1, 1, H, W, D, 3) @ (N, 1, 1, 1, 3, 3)

    def rotate_2d(self, ref, euler, out_size=None):
        # euler (B,)
        rot_ref = lie_tools.zrot(euler).unsqueeze(1) #(B, 1, 3, 3)
        #print(ref.shape, rot_ref.shape)
        #grid (1, 1, H, W, 3) x (B, 1, 3, 3) -> (1, B, H, W, 3)
        #print(self.grid2d.shape, rot_ref.shape)
        if out_size is not None:
            out_xdim = out_size//2 + 1
            head = (self.render_size - out_size)//2
            tail = head + out_size
            grid_r = self.grid2d[..., head:tail, :out_xdim, :]
        else:
            grid_r = self.grid2d

        pos_ref = grid_r @ rot_ref
        rotated_ref = F.grid_sample(ref, pos_ref[..., :2].squeeze(0), align_corners=ALIGN_CORNERS, mode='bicubic')
        return rotated_ref

    def rotate_euler(self, ref, euler):
        # ref (1, 1, z, y, x), euler (B, 2)
        Ra = lie_tools.zrot(euler[..., 0]).unsqueeze(1) #(B, 1, 3, 3)
        Rb = lie_tools.yrot(euler[..., 1]).unsqueeze(1)
        #print(ref.shape, rot_ref.shape)
        #grid (1, 1, z, y, 3) x (B, 1, 3, 3) -> (1, B, H, W, 3)
        #print(self.grid2d.shape, rot_ref.shape)
        pos = self.gridz @ Ra
        # rotate around z, sample ref (1, z, y, x)
        rotated_ref = F.grid_sample(ref.squeeze(1), pos[..., :2].squeeze(0), align_corners=ALIGN_CORNERS, mode='bicubic')
        # permute y axis to z
        rotated_ref = rotated_ref.permute(dims=[0, 2, 1, 3]) # (1, y, z, x)
        # sample ref
        pos = self.gridz @ Rb
        pos = torch.stack([pos[...,0], ps[...,2]], dim=-1)
        rotated_ref = F.grid_sample(rotated_ref, pos.squeeze(0), align_corners=ALIGN_CORNERS, mode='bicubic')
        # permute again
        return rotated_ref.permute(dims=[0, 2, 1, 3]).unsqueeze(0)

    def sample(self, src):
        return F.grid_sample(src, self.grid, align_corners=ALIGN_CORNERS)

    def pad(self, src, out_size):
        #pad the 2d output
        src_size = src.shape[-1]
        pad_size = (out_size - src_size)//2
        if pad_size == 0:
            return src
        return F.pad(src, (pad_size, pad_size, pad_size, pad_size))

    def rotate_and_sample(self, src, rot):
        pos = self.rotate(rot)
        return F.grid_sample(src, pos, align_corners=ALIGN_CORNERS)

    def forward(self, src, flow):
        # new locations
        # flow (N, 3, H, W, D)
        shape = flow.shape[2:]
        flow = flow.permute(0, 2, 3, 4, 1)
        new_locs = self.grid + flow

        # need to normalize grid values to [-1, 1] for resampler
        new_locs = 2. * (new_locs/float(self.templateres - 1) - 0.5)
        #for i in range(len(shape)):
        #    new_locs[..., i] = 2 * (new_locs[..., i] / (shape[i] - 1) - 0.5)

        return F.grid_sample(src, new_locs, align_corners=ALIGN_CORNERS, mode=self.mode)

class VanillaDecoder(nn.Module):
    def __init__(self, D, in_vol=None, Apix=1., template_type=None, templateres=256, warp_type=None, symm_group=None,
                 ctf_grid=None, fixed_deform=False, deform_emb_size=2, zdim=8, render_size=140,
                 use_fourier=False, down_vol_size=140):
        super(VanillaDecoder, self).__init__()
        self.D = D
        self.vol_size = (D - 1)
        self.Apix = Apix
        self.ctf_grid = ctf_grid
        self.template_type = template_type
        self.templateres = templateres
        self.use_conv_template = False
        self.fixed_deform = fixed_deform
        self.crop_vol_size = down_vol_size
        self.out_size = 128
        self.render_size = render_size
        self.use_fourier = use_fourier

        if symm_group is not None:
            self.symm_group = symm_groups.SymmGroup(symm_group)
            print(self.symm_group.symm_opsR[self.symm_group.SymsNo - 1])
            self.register_buffer("symm_ops_rot", torch.tensor([x.rotation_matrix for x in self.symm_group.symm_opsR]).float())
            self.register_buffer("symm_ops_trans", torch.tensor([x.translation_vector for x in self.symm_group.symm_opsR]).float())
            grid_size = self.templateres
            zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., grid_size),
                                np.linspace(-1., 1., grid_size),
                                np.linspace(-1., 1., grid_size), indexing='ij')
            #xgrid is the innermost dimension (-1, ..., 1)
            self.register_buffer("symm_grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))

        else:
            self.symm_group = None

        if self.template_type == "conv":
            self.use_conv_template = True
            self.zdim = zdim
            if self.use_fourier:
                self.template = ConvTemplate(in_dim=self.zdim, outchannels=1, templateres=self.templateres)
            else:
                self.template = ConvTemplate(in_dim=self.zdim, templateres=self.templateres)
            #self.Apix = (self.D - 1)/templateres*self.Apix
        else:
            self.template = nn.Parameter(in_vol)

        if self.use_fourier:
            zgrid, ygrid, xgrid = np.meshgrid(np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres),
                                np.linspace(-1., 1., self.templateres), indexing='ij')
            mask = torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32))
            mask = mask.pow(2).sum(-1) < 0.85 ** 2
            self.register_buffer("mask_w", mask)
            ##xgrid is the innermost dimension (-1, ..., 1)
            #self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))

            self.fourier_transformer = healpix_sampler.SpatialTransformer(self.templateres, use_fourier=True, render_size=self.render_size)
        else:
            self.transformer = SpatialTransformer(self.crop_vol_size, render_size=self.render_size)


        x_idx = torch.arange(0, self.out_size, dtype=torch.float32) - self.out_size//2
        grids = torch.meshgrid(x_idx, x_idx)
        self.register_buffer("grid2d",  torch.stack((grids[1], grids[0]), dim=-1))
        #grids[1], (0, ..., vol)
        self.warp_type = warp_type
        if self.warp_type == "blurmix":
            self.blur_kernel_size = 5
            self.blur_kernel_num = 1
            #init_blur = torch.tensor([0.5, 0., 0., 0.5, 0., 0.5])
            init_blur = torch.tensor([1.0, 0., 0., 0., 1.0, 0.])
            self.blur_rots = nn.Parameter(data=init_blur.view([2, 3]))
            self.blur_scales = nn.Parameter(data=2*torch.ones(3))
            blur_grid = utils.create_3dgrid(self.blur_kernel_size)
            blur_grid -= (self.blur_kernel_size - 1)/2
            #xgrid is the innermost dimension (-1, ..., 1)
            self.register_buffer("blur_grid", blur_grid)
            #self.blur_weight_size = self.templateres//(self.blur_kernel_size*2)
            #self.blur = AffineMixWeight(in_dim=self.zdim, out_dim=self.blur_kernel_num, out_size=self.blur_kernel_size)
            #self.blur_weight = AffineMixWeight(in_dim=self.zdim, out_dim=self.blur_kernel_num, out_size=self.blur_weight_size)
            #self.transformer_blur = SpatialTransformer(self.templateres)

        elif self.warp_type == "diffeo":
            self.flow_size = 32
            self.n_step = 2
            self.warp = AffineMixWeight(out_size=self.flow_size)
            self.transformer_diffeo = SpatialTransformer(self.flow_size, normalize=False)
            self.transformer_template = SpatialTransformer(self.templateres)

        elif self.warp_type == "deform":
            self.deform_emb_size = deform_emb_size
            self.warp = AffineMixWeight(in_dim=self.deform_emb_size)

    def translate(self, images, trans):
        B = trans.shape[0]
        trans_dim = trans.shape[1]
        grid_t = self.grid_2d + trans.view((B, 1, 1, trans_dim))
        # put in range -1, 1
        grid_t = 2.*(grid_t/float(self.crop_vol_size - 1.) - 0.5)
        #images (B, 1, H, W)
        translated = F.grid_sample(images, grid_t, align_corners=ALIGN_CORNERS, mode='bilinear')
        return translated

    def symmetrise_template(self, template, grid):
        B = template.shape[0]
        symm_template = template
        for i in range(self.symm_group.SymsNo - 1):
            pos = grid @ self.symm_ops_rot[i] + self.symm_ops_trans[i]
            pos = pos.repeat(B,1,1,1,1)
            symm_template = symm_template + F.grid_sample(template, pos, align_corners=ALIGN_CORNERS)
        return symm_template/float(self.symm_group.SymsNo + 1)

    def sample_symmetrised_ops(self, rots):
        B = rots.size(0)
        rand_choices = torch.randint(self.symm_group.SymsNo, (B,))
        symm_rots  = self.symm_ops_rot[rand_choices]
        #symm_trans = self.symm_ops_trans[rand_choices]
        symm_rots  = symm_rots @ rots
        #symm_trans = self.symm_trans @ rots
        return symm_rots

    def vecint(self, vec, bidir=False):
        #scale = 1/(2.**self.n_step)
        #vec = vec * scale #(N, 3, D, H, W)
        fields = [vec]
        if bidir:
            fields.append(-vec)
        for i in range(self.n_step):
            vec = vec + self.transformer_diffeo(vec, vec)
            fields.append(vec)
            if bidir:
                fields.append(vec)
        fields = torch.cat(fields, dim=1) #(N, 3*C, D, H, W)
        #print(fields)
        #upsample
        #fields = F.grid_sample(fields, self.grid, align_corners=True)
        fields = self.transformer_template.sample(fields)
        # permute
        #fields = fields.permute([0, 2, 3, 4, 1]) #(N, D, H, W, 3)
        return fields

    def total_variation(self, template):
        head = 1
        tail  = self.templateres
        assert head < tail
        return  torch.sqrt(1e-8 + (template[:, :, head:tail, head:tail, head:tail] - template[:, :, head:tail, head:tail, head-1:tail-1])**2 +
                  (template[:, :, head:tail, head:tail, head:tail] - template[:, :, head:tail, head-1:tail-1, head:tail])**2 +
                  (template[:, :, head:tail, head:tail, head:tail] - template[:, :, head-1:tail-1, head:tail, head:tail])**2)

    def forward(self, rots, trans, z=None, in_template=None, euler=None, ref_fft=None, ctf=None, others=None, save_mrc=False):
        if "y_fft" in others:
            others["y_fft"] = torch.view_as_complex(others["y_fft"])
        #ref_fft = torch.view_as_complex(ref_fft)
        #generate a projection
        #pos = coords.view(1, self.D, self.D, 3)
        # global rotation
        if self.use_conv_template:
            #print((z[0] == z[1]).sum())
            template = self.template(z)
        elif in_template is not None:
            template = in_template
        else:
            template = self.template.unsqueeze(0).unsqueeze(0)

        losses = {}
        if self.training:
            losses["l2"] = torch.mean((template.abs())).unsqueeze(0)
            losses["tvl2"] = torch.mean(self.total_variation(template)).unsqueeze(0) #torch.tensor(0.).to(template.get_device())

        if self.warp_type == "blurmix":
            #blur_kernel = self.blur(encoding) #(N, C, D, H, W)
            blur_rots = lie_tools.s2s2_to_SO3(self.blur_rots[0], self.blur_rots[1])
            blur_grid = self.blur_grid @ blur_rots
            blur_grid = blur_grid * self.blur_scales
            blur_grid = blur_grid.pow(2).sum(-1)
            blur_kernel = torch.exp(-blur_grid)
            print(blur_kernel.shape)
            #blur_kernel_shape = blur_kernel.shape
            #blur_kernel = F.softmax(blur_kernel.view(blur_kernel_shape[0], self.blur_kernel_num, -1),
            #                        dim=-1).view(blur_kernel_shape)
            #blur_kernel = blur_kernel.permute([1, 0, 2, 3, 4]) #(C, N, D, H, W)

            #blur_weight = self.blur_weight(encoding) #(N, C, D, H, W)
            #blur_weight = self.transformer_blur.sample(blur_weight)
            #blur_weight = F.softmax(blur_weight, dim=1)
            #losses["tvl1"] += torch.mean(self.total_variation(weight)).unsqueeze(0)

            #mulitply with weight
            template = F.conv3d(template, blur_kernel, padding='same')
            #print(template.shape, blur_kernel.shape, blur_weight.shape)
            #template = (template*blur_weight).sum(1, keepdim=True)
            #print(template.shape)

        elif self.warp_type == "diffeo":
            vec = self.warp(encoding)
            #if self.symm_group is not None:
            #    vec = self.symmetrise_template(vec, self.symm_diffeo.grid)
            fields = self.vecint(vec, bidir=True) #(N, 3*C, D, H, W)
            # now deform templates according to fields
            fields = fields.permute([0, 2, 3, 4, 1])
            #template = self.transformer.sample(template)
            templates = [template]
            num_fields = fields.shape[-1]//3
            for i in range(num_fields):
                displaced = self.transformer_template.grid + 2.*fields[:, :, :, :, 3*i:3*(i+1)]/(self.transformer_template.templateres - 1)
                template_i = F.grid_sample(template, displaced, align_corners=ALIGN_CORNERS)
                templates.append(template_i)
            template = torch.cat(templates, dim=1)

        elif self.warp_type == "deform":
            vec = self.warp(encoding) #(N, 3, D, H, W)

        if self.symm_group is not None:
            #rots = self.sample_symmetrised_ops(rots)
            template = self.symmetrise_template(template, self.symm_grid)

        if self.use_fourier:
            #mask template
            template = template * self.mask_w
            template_FT = fft.torch_rfft3_center(template)
            template_FT = template_FT[..., 1:, 1:, :self.templateres//2]
            template_FT = torch.cat((template_FT[..., 1:].flip(dims=(-1,-2,-3)).conj(), template_FT), dim=-1)
            #print(template_FT.shape)

        images = []
        refs = []
        B = rots.shape[0]
        #theta = np.zeros((3,4), dtype=np.float32)
        valid = (torch.sum(self.transformer.grid ** 2, dim=-1) < 1.).float()
        for i in range(B):
            #pos = F.affine_grid(torch.tensor(theta).unsqueeze(0), (1,1,self.vol_size,self.vol_size,self.vol_size))
            #pos = self.grid @ rots[i]#.transpose(-1, -2)
            #if not self.use_fourier:
            #    pos = self.transformer.rotate(rots[i])

            if self.warp_type == "deform":
                displacement = F.grid_sample(vec[i:i+1, ...], pos, align_corners=ALIGN_CORNERS)
                displacement = displacement.permute([0, 2, 3, 4, 1]) #(N, D, H, W, 3)
                pos = pos + 2.*displacement/(self.transformer.templateres - 1.)

            #valid = torch.prod((pos > -1.) * (pos < 1.), dim=-1).float()

            if self.fixed_deform:
                if not self.use_fourier:
                    #image = []
                    #image.append(torch.sum(vol, axis=-3).squeeze(0))
                    if "euler" in others:
                        #others_rot_i = others["rots"][i,...]
                        euler_i = torch.cat([euler[i:i+1,...], others["euler"][i,...]], dim=0)
                        euler2 = euler_i[:, 2]
                        rot = lie_tools.euler_to_SO3(euler_i[:,:2]).unsqueeze(1).unsqueeze(1)
                        ref_i = torch.cat([ref_fft[i,...], others["y"][i,...]], dim=0).unsqueeze(1)
                        #rot = torch.cat([rots[i].unsqueeze(0), others_rot_i], dim=0).unsqueeze(1).unsqueeze(1) #(1, 1, 1, 3, 3)
                        template_i = template[i:i+1,...].repeat(rot.shape[0], 1, 1, 1, 1)
                    else:
                        euler_i = euler[i:i+1,...] #(B, 3)
                        euler2 = euler_i[:, 2] #(B)
                        rot = lie_tools.euler_to_SO3(euler_i[...,:2]).unsqueeze(1).unsqueeze(1) #(B, 1, 1, 3, 3)
                        if "rot_dev" in others:
                            rot = rot @ others["rot_dev"][i, ...]
                        #print(euler_i.shape, euler2.shape, rot.shape)
                        #rot = rots[i].unsqueeze(0).unsqueeze(0).unsqueeze(0) #(1, 1, 1, 3, 3)
                        template_i = template[i:i+1,...]
                        ref_i = ref_fft[i:i+1, ...]
                        #for j in range(others_rot_i.shape[0]):
                        #    pos = self.transformer.rotate(others_rot_i[j])
                        #    image_j = F.grid_sample(template[i:i+1,...], pos, align_corners=ALIGN_CORNERS)
                        #    image_j *= valid
                        #    image.append(torch.sum(image_j, axis=-3).squeeze(0))
                    #print(rot.shape, others["rot_dev"].shape)
                    # rotate reference
                    ref = self.transformer.rotate_2d(ref_i, -euler2)
                    refs.append(ref.squeeze(1))
                    pos = self.transformer.rotate(rot) #(B, 1, H, W, D, 3) x ( B, 1, 1, 3, 3) -> (B, 1, H, W, D, 3)
                    vol = F.grid_sample(template_i, pos, align_corners=ALIGN_CORNERS)
                    #print(vol.shape, pos.shape)
                    vol *= valid
                    image = torch.sum(vol, axis=-3).squeeze(1)
                else:
                    #image = self.fourier_transformer.rotate_and_sampleFT(template_FT[i:i+1,...],
                    #                                                     rots[i], image_ori).squeeze(0)

                    #image, _ = self.fourier_transformer.rotate_and_sample_euler(template_FT[i:i+1,...],
                    #                                                            euler[i:i+1,...], ref_fft[i:i+1,...])
                    #image, ref = self.fourier_transformer.hp_sample_(template_FT[i:i+1,...],
                    #                                           euler, ref_fft, ctf,
                    #                                           trans=trans)
                    if others is not None:
                        # concatenate ref, euler, ctf, trans with others
                        euler_i = torch.cat([euler[i:i+1,...], others["euler"][i,...]], dim=0)
                        trans_i = torch.cat([trans[i:i+1,...], others["trans"][i,...]], dim=0)
                        #print(euler_i.shape, others["euler"].shape, trans_i.shape, ref_fft.shape, others["y_fft"].shape)
                        ref_fft_i = torch.cat([ref_fft[i,...], others["y_fft"][i,...]], dim=0)
                    else:
                        euler_i = euler[i:i+1,...]
                        trans_i = trans[i:i+1,...]
                        ref_fft_i = ref_fft[i,...]
                    image, ref = self.fourier_transformer.hp_sample_(template_FT[i:i+1,...],
                                                               euler_i, ref_fft_i, ctf[i:i+1,...],
                                                               trans=trans_i)

                    image = image.squeeze(0)
                    ref = ref.squeeze(0)
                    refs.append(ref)

            elif self.warp_type == "affine" or self.warp_type is None:
                #for j in range(template.shape[0]):
                #template (num_struct, 1, H, W, D) -> (1, num_struct, H, W, D)
                if self.use_fourier:
                    image, ref = self.fourier_transformer.hp_sample_(template_FT,
                                                               euler[i:i+1,...], ref_fft[i:i+1,...], ctf[i:i+1,...],
                                                               trans=trans[i:i+1,...])
                    image = image.squeeze(0) #(1, B, H, W)
                    ref = ref.squeeze(0) #(1, B, H, W)
                    refs.append(ref)
                else:
                    vol = F.grid_sample(template.squeeze(1).unsqueeze(0), pos, align_corners=ALIGN_CORNERS)
                    vol *= valid
                    image = torch.sum(vol, axis=-3)
                    image = image.squeeze(0)

            elif self.warp_type == "diffeo":
                #template (1, C, H, W, D)
                vol = F.grid_sample(template, pos, align_corners=ALIGN_CORNERS)
                vol *= valid
                image = torch.sum(vol, axis=-3)
                image = image.squeeze(0)
            #if self.warp_type == "diffeo":
            #    image_b = [image]
            #    len_disp = displacement.shape[-1]//3
            #    for j in range(len_disp):
            #        displaced = pos + 2.*displacement[:, :, :, :, 3*j:3*(j+1)]/(self.crop_vol_size - 1)
            #        vol = F.grid_sample(template.squeeze(1).unsqueeze(0), displaced, align_corners=True)
            #        vol *= valid
            #        image = torch.sum(vol, axis=-3)
            #        image = image.squeeze(0)
            #        image_b.append(image)
            #    image_b = torch.cat(image_b, dim=0)
            #    images.append(image_b)
            elif self.warp_type == "deform" or self.warp_type == "blurmix":
                vol = F.grid_sample(template, pos, align_corners=ALIGN_CORNERS)
                vol *= valid
                image = torch.sum(vol, axis=-3)
                image = image.squeeze(0)
            else:
                raise RuntimeError
            images.append(image)
        images = torch.stack(images, 0)
        refs   = torch.stack(refs, 0)
        #print(images.shape, refs.shape)
        # pad to original size
        #if not self.use_fourier:
        #    images = self.transformer.pad(images, self.render_size)
        if save_mrc:
            if self.use_fourier:
                self.save_mrc(template_FT[0:1, ...], 'reference', flip=True)
            else:
                self.save_mrc(template[0:1, ...], 'reference', flip=False)
        return {"y_recon": images, "losses": losses, "y_ref": refs}

    def save_mrc(self, template, filename, flip=False):
        with torch.no_grad():
            dev_id = template.get_device()
            if self.use_fourier:
                #template_FT = fft.torch_rfft3_center(template)
                #the origin is at self.templateres//2 - 1
                start = (self.templateres - self.vol_size)//2 - 1
                template_FT = template[..., start:start+self.vol_size, start:start+self.vol_size, \
                                      self.templateres//2-1:self.templateres//2+self.vol_size//2]
                template_FT = template_FT*(self.vol_size/self.templateres)**3
                #print(template_FT.shape)
                template = fft.torch_irfft3_center(template_FT)
            elif self.transformer.templateres != self.vol_size:
                template = self.transformer.sample(template)
            template = template.squeeze(0).squeeze(0)
            if flip:
                template = template.flip(0)
            mrc.write(filename + str(dev_id) + ".mrc", template.detach().cpu().numpy(), Apix=self.Apix, is_vol=True)

    @torch.no_grad()
    def save(self, filename, z=None, encoding=None, flip=False, Apix=1.):
        if self.template_type == "conv":
            template = self.template(z)
            if self.transformer.templateres != self.vol_size:
                #resample
                #template = F.grid_sample(template, self.grid, align_corners=True)
                template = self.transformer.sample(template)
            template = template.squeeze(0).squeeze(0)
        else:
            template = self.template
        if flip:
            template = template.flip(0)
        mrc.write(filename + ".mrc", template.detach().cpu().numpy(), Apix=Apix, is_vol=True)

    def get_vol(self, z=None):
        if self.template_type == "conv":
            template = self.template(z)
            if self.transformer.templateres != self.vol_size:
                #resample
                template = self.transformer.sample(template)
        else:
            template = self.template
        return template

    def get_images(self, z, rots, trans):
        if self.template_type == "conv":
            template = self.template(z)
        else:
            template = self.template
        B = rots.shape[0]
        images = []
        for i in range(B):
            pos = self.transformer.rotate(rots[i])
            valid = (torch.sum(pos ** 2, dim=-1) < 1.).float()

            vol = F.grid_sample(template, pos, align_corners=ALIGN_CORNERS)
            vol *= valid
            image = torch.sum(vol, axis=-3)
            image = image.squeeze(0)
            images.append(image)
        images = torch.stack(images, 0)
        if self.transformer.templateres != self.vol_size:
            images = self.transformer.pad(images, self.vol_size)
        images = self.translate(images, trans)
        return images


class PositionalDecoder(nn.Module):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf', enc_dim=None):
        super(PositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.enc_type = enc_type
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 1, activation)

    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
        if self.enc_type == 'geom_ft':
            freqs = self.DD*np.pi*(2./self.DD)**(freqs/(self.enc_dim-1)) # option 1: 2/D to 1
        elif self.enc_type == 'geom_full':
            freqs = self.DD*np.pi*(1./self.DD/np.pi)**(freqs/(self.enc_dim-1)) # option 2: 2/D to 2pi
        elif self.enc_type == 'geom_lowf':
            freqs = self.D2*(1./self.D2)**(freqs/(self.enc_dim-1)) # option 3: 2/D*2pi to 2pi
        elif self.enc_type == 'geom_nohighf':
            freqs = self.D2*(2.*np.pi/self.D2)**(freqs/(self.enc_dim-1)) # option 4: 2/D*2pi to 1
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError('Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2+1, dtype=torch.float)
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, coords):
        '''Input should be coordinates from [-.5,.5]'''
        assert (coords[...,0:3].abs() - 0.5 < 1e-4).all()
        return self.decoder(self.positional_encoding_geom(coords))

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        # Note: extent should be 0.5 by default, except when a downsampled
        # volume is generated
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent,extent,D,endpoint=True,dtype=np.float32)):
            x = coords + torch.tensor([0,0,dz])
            if zval is not None:
                x = torch.cat((x,z), dim=-1)
            with torch.no_grad():
                y = self.forward(x)
                y = y.view(D,D).cpu().numpy()
            vol_f[i] = y
        vol_f = vol_f*norm[1]+norm[0]
        vol = fft.ihtn_center(vol_f[0:-1,0:-1,0:-1]) # remove last +k freq for inverse FFT
        return vol

class FTPositionalDecoder(nn.Module):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, enc_type='linear_lowf', enc_dim=None):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_type = enc_type
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.in_dim = 3 * (self.enc_dim) * 2 + self.zdim
        self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 2, activation)

    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
        if self.enc_type == 'geom_ft':
            freqs = self.DD*np.pi*(2./self.DD)**(freqs/(self.enc_dim-1)) # option 1: 2/D to 1
        elif self.enc_type == 'geom_full':
            freqs = self.DD*np.pi*(1./self.DD/np.pi)**(freqs/(self.enc_dim-1)) # option 2: 2/D to 2pi
        elif self.enc_type == 'geom_lowf':
            freqs = self.D2*(1./self.D2)**(freqs/(self.enc_dim-1)) # option 3: 2/D*2pi to 2pi
        elif self.enc_type == 'geom_nohighf':
            freqs = self.D2*(2.*np.pi/self.D2)**(freqs/(self.enc_dim-1)) # option 4: 2/D*2pi to 1
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError('Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2+1, dtype=torch.float)
        freqs = freqs.view(*[1]*len(coords.shape), -1) # 1 x 1 x D2
        coords = coords.unsqueeze(-1) # B x 3 x 1
        k = coords[...,0:3,:] * freqs # B x 3 x D2
        s = torch.sin(k) # B x 3 x D2
        c = torch.cos(k) # B x 3 x D2
        x = torch.cat([s,c], -1) # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim-self.zdim) # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x,coords[...,3:,:].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def forward(self, lattice):
        '''
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        '''
        # if ignore_DC = False, then the size of the lattice will be odd (since it
        # includes the origin), so we need to evaluate one additional pixel
        c = lattice.shape[-2]//2 # top half
        cc = c + 1 if lattice.shape[-2] % 2 == 1 else c # include the origin
        assert abs(lattice[...,0:3].mean()) < 1e-4, '{} != 0.0'.format(lattice[...,0:3].mean())
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[...,0:cc,:])
        image[..., 0:cc] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,cc:] = (top_half[...,0] + top_half[...,1])[...,np.arange(c-1,-1,-1)]
        return image

    def decode(self, lattice):
        '''Return FT transform'''
        assert (lattice[...,0:3].abs() - 0.5 < 1e-4).all()
        # convention: only evalute the -z points
        w = lattice[...,2] > 0.0
        lattice[...,0:3][w] = -lattice[...,0:3][w] # negate lattice coordinates where z > 0
        result = self.decoder(self.positional_encoding_geom(lattice))
        result[...,1][w] *= -1 # replace with complex conjugate to get correct values for original lattice positions
        return result

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        assert extent <= 0.5
        if zval is not None:
            zdim = len(zval)
            z = torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent,extent,D,endpoint=True,dtype=np.float32)):
            x = coords + torch.tensor([0,0,dz])
            keep = x.pow(2).sum(dim=1) <= extent**2
            x = x[keep]
            if zval is not None:
                x = torch.cat((x,z.expand(x.shape[0],zdim)), dim=-1)
            with torch.no_grad():
                if dz == 0.0:
                    y = self.forward(x)
                else:
                    y = self.decode(x)
                    y = y[...,0] - y[...,1]
                slice_ = torch.zeros(D**2, device='cpu')
                slice_[keep] = y.cpu()
                slice_ = slice_.view(D,D).numpy()
            vol_f[i] = slice_
        vol_f = vol_f*norm[1]+norm[0]
        vol = fft.ihtn_center(vol_f[:-1,:-1,:-1]) # remove last +k freq for inverse FFT
        return vol

class FTSliceDecoder(nn.Module):
    '''
    Evaluate a central slice out of a 3D FT of a model, returns representation in
    Hartley reciprocal space

    Exploits the symmetry of the FT where F*(x,y) = F(-x,-y) and only
    evaluates half of the lattice. The decoder is f(x,y,z) => real, imag
    '''
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation):
        '''D: image width or height'''
        super(FTSliceDecoder, self).__init__()
        self.decoder = ResidLinearMLP(in_dim, nlayers, hidden_dim, 2, activation)
        D2 = int(D/2)

        ### various pixel indices to keep track of for forward_even
        self.center = D2*D + D2
        self.extra = np.arange((D2+1)*D, D**2, D) # bottom-left column without conjugate pair
        # evalute the top half of the image up through the center pixel
        # and extra bottom-left column (todo: just evaluate a D-1 x D-1 image so
        # we don't have to worry about this)
        self.all_eval = np.concatenate((np.arange(self.center+1), self.extra))

        # pixel indices for the top half of the image up to (but not incl)
        # the center pixel and excluding the top row and left-most column
        i, j = np.meshgrid(np.arange(1,D),np.arange(1,D2+1))
        self.top = (j*D+i).ravel()[:-D2]

        # pixel indices for bottom half of the image after the center pixel
        # excluding left-most column and given in reverse order
        i, j =np.meshgrid(np.arange(1,D),np.arange(D2,D))
        self.bottom_rev = (j*D+i).ravel()[D2:][::-1].copy()

        self.D = D
        self.D2 = D2

    def forward(self, lattice):
        '''
        Call forward on central slices only
            i.e. the middle pixel should be (0,0,0)

        lattice: B x N x 3+zdim
        '''
        assert lattice.shape[-2] % 2 == 1
        c = lattice.shape[-2]//2 # center pixel
        assert lattice[...,c,0:3].sum() == 0.0, '{} != 0.0'.format(lattice[...,c,0:3].sum())
        assert abs(lattice[...,0:3].mean()) < 1e-4, '{} != 0.0'.format(lattice[...,0:3].mean())
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[...,0:c+1,:])
        image[..., 0:c+1] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,c+1:] = (top_half[...,0] + top_half[...,1])[...,np.arange(c-1,-1,-1)]
        return image

    def forward_even(self, lattice):
        '''Extra bookkeeping with extra row/column for an even sized DFT'''
        image = torch.empty(lattice.shape[:-1])
        top_half = self.decode(lattice[...,self.all_eval,:])
        image[..., self.all_eval] = top_half[...,0] - top_half[...,1]
        # the bottom half of the image is the complex conjugate of the top half
        image[...,self.bottom_rev] = top_half[...,self.top,0] + top_half[...,self.top,1]
        return image

    def decode(self, lattice):
        '''Return FT transform'''
        # convention: only evalute the -z points
        w = lattice[...,2] > 0.0
        lattice[...,0:3][w] = -lattice[...,0:3][w] # negate lattice coordinates where z > 0
        result = self.decoder(lattice)
        result[...,1][w] *= -1 # replace with complex conjugate to get correct values for original lattice positions
        return result

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume

        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D**2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D,D,D),dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent,extent,D,endpoint=True,dtype=np.float32)):
            x = coords + torch.tensor([0,0,dz])
            if zval is not None:
                x = torch.cat((x,z), dim=-1)
            with torch.no_grad():
                y = self.decode(x)
                y = y[...,0] - y[...,1]
                y = y.view(D,D).cpu().numpy()
            vol_f[i] = y
        vol_f = vol_f*norm[1]+norm[0]
        vol_f = utils.zero_sphere(vol_f)
        vol = fft.ihtn_center(vol_f[:-1,:-1,:-1]) # remove last +k freq for inverse FFT
        return vol

class VAE(nn.Module):
    def __init__(self,
            lattice,
            qlayers, qdim,
            players, pdim,
            encode_mode = 'mlp',
            no_trans = False,
            enc_mask = None
            ):
        super(VAE, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        self.in_dim = lattice.D*lattice.D if enc_mask is None else enc_mask.sum()
        self.enc_mask = enc_mask
        assert qlayers > 2
        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, qdim)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(self.in_dim,
                            qlayers-2, # -2 bc we add 2 more layers in the homeomorphic encoer
                            qdim,  # hidden_dim
                            qdim, # out_dim
                            nn.ReLU) #in_dim -> hidden_dim
        elif encode_mode == 'mlp':
            self.encoder = MLP(self.in_dim,
                            qlayers-2,
                            qdim, # hidden_dim
                            qdim, # out_dim
                            nn.ReLU) #in_dim -> hidden_dim
        else:
            raise RuntimeError('Encoder mode {} not recognized'.format(encode_mode))
        # predict rotation and translation in two completely separate NNs
        #self.so3_encoder = SO3reparameterize(qdim) # hidden_dim -> SO(3) latent variable
        #self.trans_encoder = ResidLinearMLP(nx*ny, 5, qdim, 4, nn.ReLU)

        # or predict rotation/translations from intermediate encoding
        self.so3_encoder = SO3reparameterize(qdim, 1, qdim) # hidden_dim -> SO(3) latent variable
        self.trans_encoder = ResidLinearMLP(qdim, 1, qdim, 4, nn.ReLU)

        self.decoder = FTSliceDecoder(3, self.D, players, pdim, nn.ReLU)
        self.no_trans = no_trans

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def encode(self, img):
        '''img: BxDxD'''
        img = img.view(img.size(0),-1)
        if self.enc_mask is not None:
            img = img[:,self.enc_mask]
        enc = nn.ReLU()(self.encoder(img))
        z_mu, z_std = self.so3_encoder(enc)
        if self.no_trans:
            tmu, tlogvar = (None, None)
        else:
            z = self.trans_encoder(enc)
            tmu, tlogvar = z[:,:2], z[:,2:]
        return z_mu, z_std, tmu, tlogvar

    def eval_volume(self, norm):
        return self.decoder.eval_volume(self.lattice.coords, self.D, self.lattice.extent, norm)

    def decode(self, rot):
        # transform lattice by rot.T
        x = self.lattice.coords @ rot # R.T*x
        y_hat = self.decoder(x)
        y_hat = y_hat.view(-1, self.D, self.D)
        return y_hat

    def forward(self, img):
        z_mu, z_std, tmu, tlogvar = self.encode(img)
        rot, w_eps = self.so3_encoder.sampleSO3(z_mu, z_std)
        # transform lattice by rot and predict image
        y_hat = self.decode(rot)
        if not self.no_trans:
            # translate image by t
            B = img.size(0)
            t = self.reparameterize(tmu, tlogvar)
            t = t.unsqueeze(1) # B x 1 x 2
            img = self.lattice.translate_ht(img.view(B,-1), t)
            img = img.view(B,self.D, self.D)
        return y_hat, img, z_mu, z_std, w_eps, tmu, tlogvar

class TiltVAE(nn.Module):
    def __init__(self,
            lattice, tilt,
            qlayers, qdim,
            players, pdim,
            no_trans=False,
            enc_mask=None
            ):
        super(TiltVAE, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        self.in_dim = lattice.D*lattice.D if enc_mask is None else enc_mask.sum()
        self.enc_mask = enc_mask
        assert qlayers > 3
        self.encoder = ResidLinearMLP(self.in_dim,
                                      qlayers-3,
                                      qdim,
                                      qdim,
                                      nn.ReLU)
        self.so3_encoder = SO3reparameterize(2*qdim, 3, qdim) # hidden_dim -> SO(3) latent variable
        self.trans_encoder = ResidLinearMLP(2*qdim, 2, qdim, 4, nn.ReLU)
        self.decoder = FTSliceDecoder(3, self.D, players, pdim, nn.ReLU)
        assert tilt.shape == (3,3), 'Rotation matrix input required'
        self.tilt = torch.tensor(tilt)
        self.no_trans = no_trans

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def eval_volume(self, norm):
        return self.decoder.eval_volume(self.lattice.coords, self.D, self.lattice.extent, norm)

    def encode(self, img, img_tilt):
        img = img.view(img.size(0), -1)
        img_tilt = img_tilt.view(img_tilt.size(0), -1)
        if self.enc_mask is not None:
            img = img[:,self.enc_mask]
            img_tilt = img_tilt[:,self.enc_mask]
        enc1 = self.encoder(img)
        enc2 = self.encoder(img_tilt)
        enc = torch.cat((enc1,enc2), -1) # then nn.ReLU?
        z_mu, z_std = self.so3_encoder(enc)
        rot, w_eps = self.so3_encoder.sampleSO3(z_mu, z_std)
        if self.no_trans:
            tmu, tlogvar, t = (None,None,None)
        else:
            z = self.trans_encoder(enc)
            tmu, tlogvar = z[:,:2], z[:,2:]
            t = self.reparameterize(tmu, tlogvar)
        return z_mu, z_std, w_eps, rot, tmu, tlogvar, t

    def forward(self, img, img_tilt):
        B = img.size(0)
        z_mu, z_std, w_eps, rot, tmu, tlogvar, t = self.encode(img, img_tilt)
        if not self.no_trans:
            t = t.unsqueeze(1) # B x 1 x 2
            img = self.lattice.translate_ht(img.view(B,-1), -t)
            img_tilt = self.lattice.translate_ht(img_tilt.view(B,-1), -t)
            img = img.view(B, self.D, self.D)
            img_tilt = img_tilt.view(B, self.D, self.D)

        # rotate lattice by rot.T
        x = self.lattice.coords @ rot # R.T*x
        y_hat = self.decoder(x)
        y_hat = y_hat.view(-1, self.D, self.D)

        # tilt series pair
        x = self.lattice.coords @ self.tilt @ rot
        y_hat2 = self.decoder(x)
        y_hat2 = y_hat2.view(-1, self.D, self.D)
        return y_hat, y_hat2, img, img_tilt, z_mu, z_std, w_eps, tmu, tlogvar

# fixme: this is half-deprecated (not used in TiltVAE, but still used in tilt BNB)
class TiltEncoder(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(TiltEncoder, self).__init__()
        assert nlayers > 2
        self.encoder1 = ResidLinearMLP(in_dim, nlayers-2, hidden_dim, hidden_dim, activation)
        self.encoder2 = ResidLinearMLP(hidden_dim*2, 2, hidden_dim, out_dim, activation)
        self.in_dim = in_dim

    def forward(self, x, x_tilt):
        x_enc = self.encoder1(x)
        x_tilt_enc = self.encoder1(x_tilt)
        z = self.encoder2(torch.cat((x_enc,x_tilt_enc),-1))
        return z

class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(ResidLinearMLP, self).__init__()
        layers = [ResidLinear(in_dim, hidden_dim) if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(ResidLinear(hidden_dim, out_dim) if out_dim == hidden_dim else nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResidLinear(nn.Module):
    def __init__(self, nin, nout):
        super(ResidLinear, self).__init__()
        self.linear = nn.Linear(nin, nout)
        #self.linear = nn.utils.weight_norm(nn.Linear(nin, nout))

    def forward(self, x):
        z = self.linear(x) + x
        return z

class MLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(MLP, self).__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Adapted from soumith DCGAN
class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(ConvEncoder, self).__init__()
        ndf = hidden_dim
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, out_dim, 4, 1, 0, bias=False),
            # state size. out_dims x 1 x 1
        )
    def forward(self, x):
        x = x.view(-1,1,64,64)
        x = self.main(x)
        return x.view(x.size(0), -1) # flatten

class SO3reparameterize(nn.Module):
    '''Reparameterize R^N encoder output to SO(3) latent variable'''
    def __init__(self, input_dims, nlayers=None, hidden_dim=None):
        super().__init__()
        if nlayers is not None:
            self.main = ResidLinearMLP(input_dims, nlayers, hidden_dim, 9, nn.ReLU)
        else:
            self.main = nn.Linear(input_dims, 9)

        # start with big outputs
        #self.s2s2map.weight.data.uniform_(-5,5)
        #self.s2s2map.bias.data.uniform_(-5,5)

    def sampleSO3(self, z_mu, z_std):
        '''
        Reparameterize SO(3) latent variable
        # z represents mean on S2xS2 and variance on so3, which enocdes a Gaussian distribution on SO3
        # See section 2.5 of http://ethaneade.com/lie.pdf
        '''
        # resampling trick
        if not self.training:
            return z_mu, z_std
        eps = torch.randn_like(z_std)
        w_eps = eps*z_std
        rot_eps = lie_tools.expmap(w_eps)
        #z_mu = lie_tools.quaternions_to_SO3(z_mu)
        rot_sampled = z_mu @ rot_eps
        return rot_sampled, w_eps

    def forward(self, x):
        z = self.main(x)
        z1 = z[:,:3].double()
        z2 = z[:,3:6].double()
        z_mu = lie_tools.s2s2_to_SO3(z1,z2).float()
        logvar = z[:,6:]
        z_std = torch.exp(.5*logvar) # or could do softplus
        return z_mu, z_std




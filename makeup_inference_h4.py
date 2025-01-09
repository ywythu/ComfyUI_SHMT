import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from .ldmx.util import instantiate_from_config,log_txt_as_img
from .ldmx.models.diffusion.ddim_test import DDIMSampler
from .ldmx.models.diffusion.plms import PLMSSampler
from .ldmx.data.makeup_dataset import MakeupDatasetTest
import torch.nn.functional as F

# from .ldmx.util import log_txt_as_img, exists, instantiate_from_config


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)



def infer_main_h4(model, seed, width,height,latent_channels,
                     downsampling_factor,ddim_steps,ddim_eta,batch_size,scale,precision,
                  skip_save,plms,fixed_code,device,outpath,ref_image_path,ref_seg_path,source_seg_path,source_depth_path,source_image_path):

    seed_everything(seed)

    model = model.to(device)

    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    start_code = None
    if fixed_code:
        start_code = torch.randn([batch_size, latent_channels, height // downsampling_factor, width // downsampling_factor], device=device)

    dataset = MakeupDatasetTest(mode='test_pair', source_image_path=source_image_path, source_seg_path=source_seg_path,
                                source_depth_path=source_depth_path,
                                ref_image_path=ref_image_path,ref_seg_path=ref_seg_path,)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    count=0

    precision_scope = autocast if precision == "autocast" else nullcontext
    output_img = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for iter, batch in enumerate(test_loader):
                    count+=1
                    source_image = batch['source_image'].to(device).float()
                    ref_image = batch['ref_image'].to(device).float()

                    source_depth = batch['source_depth'].to(device).float()
                    source_bg= batch['source_bg'].to(device).float()
                    source_face_gray = batch['source_face_gray'].to(device).float()
                    ref_face = batch['ref_face'].to(device).float()

                    source_seg_onehot = batch['source_seg_onehot'].to(device).float()
                    ref_seg_onehot = batch['ref_seg_onehot'].to(device).float()

                    source_face_seg = batch['source_face_seg'].to(device).float()

                    # get h4
                    source_HF_0 = model.lap_pyr_c1.pyramid_decom(source_face_gray)[4] # h4
                    
                    #source_HF_0_down4 = F.interpolate(source_HF_0, size=(64,64))
                    source_HF_0_down4 = F.interpolate(source_HF_0, size=(height // downsampling_factor, width // downsampling_factor))

   
                    source_depth_down4 = F.pixel_unshuffle(source_depth, downscale_factor=4)

                    source_HF = torch.cat([source_HF_0_down4,source_depth_down4],dim=1)

                    ref_LF_64 = F.pixel_unshuffle(ref_face, downscale_factor=4)


                    #source_face_seg_64 = F.interpolate(source_face_seg, size=(64, 64), mode='bilinear')
                    source_face_seg_64 = F.interpolate(source_face_seg, size=(height // downsampling_factor, width // downsampling_factor), mode='bilinear')
                    

                    encoder_posterior_bg = model.encode_first_stage(source_bg)
                    z_bg = model.get_first_stage_encoding(encoder_posterior_bg).detach()

                    encoder_posterior_ref_LF = model.encode_first_stage(ref_face)
                    z_ref_LF = model.get_first_stage_encoding(encoder_posterior_ref_LF).detach()
                    print(z_ref_LF.shape)
                    test_model_kwargs = {}
                    test_model_kwargs['z_bg'] = z_bg.to(device)
                    test_model_kwargs['source_HF'] = source_HF.to(device)
                    test_model_kwargs['z_ref_LF'] = z_ref_LF.to(device)
                    test_model_kwargs['ref_LF_64'] = ref_LF_64.to(device)
                    test_model_kwargs['source_face_seg_64'] = source_face_seg_64.to(device)

                    test_model_kwargs['source_seg_onehot'] = source_seg_onehot.to(device)
                    test_model_kwargs['ref_seg_onehot'] = ref_seg_onehot.to(device)

                    uc = None
                    if scale != 1.0:
                        uc = model.learnable_vector
                    shape = [latent_channels,height // downsampling_factor, width // downsampling_factor]
                    samples_ddim, intermediates = sampler.sample(S=ddim_steps,
                                                     batch_size=batch_size,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta,
                                                     x_T=start_code,
                                                    log_every_t=1,
                                                     test_model_kwargs=test_model_kwargs)

                    
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    source_image = torch.clamp((source_image + 1.0) / 2.0, min=0.0, max=1.0)
                    ref_image = torch.clamp((ref_image + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    if not skip_save:
                        row_0=torch.cat([source_image[0,::],ref_image[0, ::],x_samples_ddim[0, ::]],dim=2)
                        grid = make_grid(row_0)
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        grid_path = os.path.join(outpath, "grid_" + f'scale{scale}_step{ddim_steps}')
                        
                        os.makedirs(grid_path, exist_ok=True)
                        img.save(os.path.join(grid_path, 'grid_' + '%05d'%count + '_seed_'+str(seed)+'.jpg'))
                        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
                              f" \nEnjoy.")
                    
                    output_img.append(x_samples_ddim)
    return output_img


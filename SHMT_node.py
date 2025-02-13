# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import torch
from omegaconf import OmegaConf
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from .makeup_inference_h0 import load_model_from_config as load_model_from_config
from .makeup_inference_h0 import infer_main_h0
from .makeup_inference_h4 import infer_main_h4
from .makeup_inference_model2 import infer_main_v2
from .node_utils import pre_image_data

import folder_paths

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))

weigths_SHMT_current_path = os.path.join(folder_paths.models_dir, "SHMT")
if not os.path.exists(weigths_SHMT_current_path):
    os.makedirs(weigths_SHMT_current_path)

try:
    folder_paths.add_model_folder_path("SHMT", weigths_SHMT_current_path, False)
except:
    folder_paths.add_model_folder_path("SHMT", weigths_SHMT_current_path)


class SHMT_LoadModel:
    def __init__(self):
        self.model = None
        self.mode = None
        self.image_processor = None
        self.seg_model = None
        self.model_name = None
        self.model2 = None

    @classmethod
    def INPUT_TYPES(cls):
        shmt_ckpt_list = [i for i in folder_paths.get_filename_list("SHMT") if
                          "epoch" in i]
        return {
            "required": {
                "shmt_ckpt": (["none"] + shmt_ckpt_list,),
                "ldm_ae": (["none"] + folder_paths.get_filename_list("SHMT"),),
                "face_repo": ("STRING", {"default": "F:/ComfyUI311/ComfyUI/models/SHMT/face-parsing"}),
                "enable_model2": ("BOOLEAN", {"default": False},),
            }
        }
    
    RETURN_TYPES = ("MODEL_SHMT",)
    RETURN_NAMES = ("model",)
    FUNCTION = "main_loader"
    CATEGORY = "SHMT"
    
    def main_loader(self, shmt_ckpt, ldm_ae, face_repo, enable_model2):
        if ldm_ae != "none":
            ae_ckpt = folder_paths.get_full_path("SHMT", ldm_ae)
        else:
            raise "need choice ae ckpt"
        model2 = None
        if shmt_ckpt != "none":
            if "710" in shmt_ckpt:  # h4
                config = OmegaConf.load(os.path.join(current_path, "configs/latent-diffusion/shmt_h4.yaml"))
                config.model.params.first_stage_config.params.ckpt_path = ae_ckpt
                if not self.model or self.mode != "h4":
                    self.model = load_model_from_config(config, folder_paths.get_full_path("SHMT", shmt_ckpt))
                if enable_model2:
                    print("***********infer mix model************")
                    config2 = OmegaConf.load(os.path.join(current_path, "configs/latent-diffusion/shmt_h0.yaml"))
                    config2.model.params.first_stage_config.params.ckpt_path = ae_ckpt

                    self.model2 = load_model_from_config(config2,
                                                    os.path.join(folder_paths.models_dir, "SHMT/epoch=000755-001.ckpt"))
                    self.mode = "mix"
                else:
                    self.mode = "h4"
                    print("***********infer h4 model************")
            elif "755" in shmt_ckpt:  # h0:
                config = OmegaConf.load(os.path.join(current_path, "configs/latent-diffusion/shmt_h0.yaml"))
                config.model.params.first_stage_config.params.ckpt_path = ae_ckpt
                if enable_model2:
                    print("***********infer mix model************")
                    self.mode = "mix"
                    config1 = OmegaConf.load(os.path.join(current_path, "configs/latent-diffusion/shmt_h4.yaml"))
                    config1.model.params.first_stage_config.params.ckpt_path = ae_ckpt
                    self.model = load_model_from_config(config1,
                                                   os.path.join(folder_paths.models_dir, "SHMT/epoch=000710-001.ckpt"))
                    self.model2 = load_model_from_config(config, folder_paths.get_full_path("SHMT", shmt_ckpt))
                else:
                    print("***********infer h0 model************")
                    if not self.model or self.mode != "h0":
                        self.mode = "h0"
                        self.model = load_model_from_config(config, folder_paths.get_full_path("SHMT", shmt_ckpt))
            else:
                raise "h4 ckpt need 755 ,h0 need 710 number in name,do'not rename it. "
        else:
            raise "need choice ckpt"
        
        # pre seg
        if face_repo and not (self.image_processor and self.seg_model):
            self.image_processor = SegformerImageProcessor.from_pretrained(face_repo)
            self.seg_model = SegformerForSemanticSegmentation.from_pretrained(face_repo)
        else:
            raise "Need fill 'jonathandinu/face-parsing' or local folder  "
        
        return (
        {"model": self.model, "model2": self.model2, "mode": self.mode, "image_processor": self.image_processor, "seg_model": self.seg_model},)


class SHMT_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "id_image": ("IMAGE",),  # [B,H,W,C], C=3
                "makeup_image": ("IMAGE",),  # [B,H,W,C], C=3
                "model": ("MODEL_SHMT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": MAX_SEED}),
                "width": ("INT", {"default": 256, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 256, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "latent_channels": ("INT", {"default": 3, "min": 3, "max": 4}),
                "downsampling_factor": ("INT", {"default": 4, "min": 2, "max": 64}),
                "ddim_steps": ("INT", {"default": 50, "min": 1, "max": 4096, "step": 1, "display": "number"}),
                "ddim_eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "proportionOFmodel1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "precision": (["autocast", "full"],),
                "skip_save": ("BOOLEAN", {"default": True},),
                "plms": ("BOOLEAN", {"default": False},),
                "fixed_code": ("BOOLEAN", {"default": True},),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sampler_main"
    CATEGORY = "SHMT"
    
    def sampler_main(self, id_image, makeup_image, model, seed, width, height,latent_channels,
                     downsampling_factor, ddim_steps, ddim_eta, batch_size, scale, precision,
                     skip_save, plms, fixed_code, proportionOFmodel1):
        mode = model.get("mode")
        model2 = model.get("model2")
        seg_model = model.get("seg_model")
        image_processor = model.get("image_processor")
        model = model.get("model")
        
        # pre img and data
        print("******** pre  input image,get seg ,3d map *********")
        ref_image_path, ref_seg_path, source_depth_path, source_seg_path, source_image_path = (
            pre_image_data(id_image,makeup_image,seg_model,image_processor,width,height))
        
        print("******** start infer *********")
        if mode == "h4":
            output_img = infer_main_h4(model, seed, width, height, latent_channels,
                                       downsampling_factor, ddim_steps, ddim_eta, batch_size, scale, precision,
                                       skip_save, plms, fixed_code, device, folder_paths.get_output_directory(),
                                       ref_image_path, ref_seg_path, source_seg_path,
                                       source_depth_path, source_image_path)
        elif model == "mix":#has bug
            output_img = infer_main_v2(model, model2, seed, width, height, latent_channels,
                                       downsampling_factor, ddim_steps, ddim_eta, batch_size, scale, precision,
                                       skip_save, plms, fixed_code, device, folder_paths.get_output_directory(),
                                       ref_image_path, ref_seg_path, source_seg_path,
                                       source_depth_path, source_image_path, proportionOFmodel1)
        else:
            output_img = infer_main_h0(model, seed, width, height,  latent_channels,
                                       downsampling_factor, ddim_steps, ddim_eta, batch_size,  scale, precision,
                                       skip_save, plms, fixed_code, device, folder_paths.get_output_directory(),
                                       ref_image_path, ref_seg_path, source_seg_path,
                                       source_depth_path, source_image_path)
        
        # model.to("cpu")#显存不会自动释放，手动迁移，不然很容易OOM
        # torch.cuda.empty_cache()
        image = output_img[0].permute(0, 2, 3, 1) if len(output_img) == 1 else\
            torch.cat(output_img, dim=0).permute(0,2,3,1)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "SHMT_LoadModel": SHMT_LoadModel,
    "SHMT_Sampler": SHMT_Sampler
    
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SHMT_LoadModel": "SHMT_LoadModel",
    "SHMT_Sampler": "SHMT_Sampler",
}

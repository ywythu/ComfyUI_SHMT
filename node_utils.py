# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
from PIL import Image
import numpy as np
import cv2
import random
from torch import nn
from omegaconf import OmegaConf
from comfy.utils import common_upscale,ProgressBar
from .THREEDDFA_V2.demo import main as three_ddfa
import folder_paths

weight_dtype = torch.float16
cur_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def pre_image_data(id_image, makeup_image,seg_model,image_processor,width,height):
    #tensor to image
    source_image = nomarl_upscale(id_image,width,height)
    ref_image = nomarl_upscale(makeup_image,width,height)
    #make dir
    folder_prefix = ''.join(random.choice("0123456789") for _ in range(5))
    infer_folder = os.path.join(folder_paths.get_temp_directory(), f"temp_{folder_prefix}")
    
    # save base image
    ref_image_path = os.path.join(infer_folder, f"infer_{folder_prefix}")
    os.makedirs(ref_image_path, exist_ok=True)
    ref_image.save(os.path.join(ref_image_path, f"{folder_prefix}.png"))
   
    source_image_path = os.path.join(infer_folder, f"source_{folder_prefix}")
    os.makedirs(source_image_path, exist_ok=True)
    source_image.save(os.path.join(source_image_path, f"{folder_prefix}.png"))
    
    # get image's seg
    seg_model.to(device)
    ref_seg_path = os.path.join(infer_folder, f"seg_{folder_prefix}")
    os.makedirs(ref_seg_path, exist_ok=True)
    ref_seg_img = face_parsing_main(ref_image.copy(), seg_model, image_processor,True)
    cv2.imwrite(os.path.join(ref_seg_path, f"{folder_prefix}.png"), ref_seg_img)
    
    source_seg_path = os.path.join(infer_folder, f"source_seg_{folder_prefix}")
    os.makedirs(source_seg_path, exist_ok=True)
    source_seg_img=face_parsing_main(source_image.copy(), seg_model, image_processor,False)
    cv2.imwrite(os.path.join(source_seg_path, f"{folder_prefix}.png"), source_seg_img)
    
    #pre 3d
    source_depth_path = os.path.join(infer_folder, f"source_dep_{folder_prefix}")
    os.makedirs(source_depth_path, exist_ok=True)
    
    config_depth = {"config":os.path.join(cur_path,"THREEDDFA_V2/configs/resnet_120x120.yml"),"folder_prefix":folder_prefix,"checkpoint_fp":os.path.join(folder_paths.models_dir,"SHMT/resnet22.pth"),
                    "bfm_fp":os.path.join(cur_path,"THREEDDFA_V2/configs/bfm_noneck_v3.pkl"),"with_bg_flag":False,"onnx":False,"img_fp":os.path.join(source_image_path, f"{folder_prefix}.png"),"opt":"3d","mode":"gpu","show_flag":False,"folder":source_depth_path}
    config = OmegaConf.create(config_depth)
    three_ddfa(config)
    
    return ref_image_path, ref_seg_path, source_depth_path, source_seg_path, source_image_path


def face_parsing_main(image,seg_model,image_processor,if_makeup):
    # run inference on image
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = seg_model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)
    
    # resize output to match input image dimensions
    upsampled_logits = nn.functional.interpolate(logits,size=image.size[::-1],  # H x W
                                                 mode='bilinear',
                                                 align_corners=False) #torch.Size([1, 19, 512, 512])
    
    color_map = {0: [0, 0, 0], 1: [1, 1, 1], 2: [2, 2, 2], 3: [3, 3, 3], 4: [4, 4, 4], 5: [5, 5, 5], 6: [6, 6, 6],
                 7: [7, 7, 7],8: [8, 8, 8], 9: [9, 9, 9], 10: [10, 10, 10], 11: [11, 11, 11], 12: [12, 12, 12],
                 13: [13, 13, 13], 14: [14, 14, 14],15: [15, 15, 15], 16: [16, 16, 16], 17: [17, 17, 17], 18: [18, 18, 18]}
    #eye-glass,neck_lin,era rin
    label_map= {"0": "background","1": "skin","2": "nose","3": "eye_g","4": "l_eye","5": "r_eye","6": "l_brow",
                "7": "r_brow","8": "l_ear","9": "r_ear","10": "mouth","11": "u_lip","12": "l_lip","13": "hair",
                "14": "hat","15": "ear_r","16": "neck_l","17": "neck","18": "cloth"},
    # get label masks
    labels = upsampled_logits.argmax(dim=1)[0]  # tensor torch.Size([512, 512])
    labels_viz = labels.cpu().numpy()  # numpy.ndarray [512, 512# ]
    
    # unique_labels = np.unique(labels_viz)[0,..,18]
    
    height, width = labels_viz.shape
    cv_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            label = labels_viz[i, j]
            cv_image[i, j] = color_map[label]
    
    #ret, thresholded_image = cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY)

    return cv_image


def pil2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = pil2narry(value)
        list_in[i] = modified_value
    return list_in


def gen_img_form_video(tensor):
    pil = []
    for x in tensor:
        pil[x] = tensor_to_pil(x)
    yield pil


def phi_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        list_in[i] = value
    return list_in

def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil

def tensor_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples


def tensor2cv(tensor_image):
    if len(tensor_image.shape)==4:# b hwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def cvargb2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def cv2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def images_generator(img_list: list,):
    #get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_,Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_,np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in=img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in,np.ndarray):
            i=cv2.cvtColor(img_in,cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            #print(i.shape)
            return i
        else:
           raise "unsupport image list,must be pil,cv2 or tensor!!!"
        
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image

def load_images(img_list: list,):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images

def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def cf_tensor2cv(tensor,width, height):
    d1, _, _, _ = tensor.size()
    if d1 > 1:
        tensor_list = list(torch.chunk(tensor, chunks=d1))
        tensor = [tensor_list][0]
    cr_tensor=tensor_upscale(tensor,width, height)
    cv_img=tensor2cv(cr_tensor)
    return cv_img

def tensor2pillist(tensor):
    b, _, _, _ = tensor.size()
    if b == 1:
       img_list = [nomarl_upscale(tensor, 768, 768)]
    else:
        image_= torch.chunk(tensor, chunks=b)
        img_list = [nomarl_upscale(i, 768, 768) for i in image_]  # pil
    return img_list
    

  

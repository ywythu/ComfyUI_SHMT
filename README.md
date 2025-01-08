# ComfyUI_SHMT
You can use [SHMT](https://github.com/Snowfallingplum/SHMT) method to apply makeup to the characters  when use ComfyUI 

# Update 
* 插件测试环境，torch2.5.1 ，cuda124，python311 / Node testing environment, torch2.5.1 ，cuda124，python311
* 目前模型mix无法使用，plms模式无法使用，出图尺寸修改了原方法只能跑256x256的限制，12G Vram跑640x640有几率OOM；请务必用正方形图片； 
* At present, the model 'mix' cannot be used, the 'plms' mode cannot be used, and the drawing size has been modified to the original method's limit of only running 256 x 256. There is a chance of OOM when running 640 x 640 in 12G Vram，Please make sure to use a square image；  


# 1. Installation

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_SHMT.git
```
---
  
# 2. Requirements  

```
pip install -r requirements.txt
```
# 3. Install Notice 安装难点 
Two methods need to be manually compiled/需要手动编译2个方法  
* 3.1 open "THREEDDFA_V2\Sim3DR\readme.md" and Follow the instructions inside to compile;  
* 打开"THREEDDFA_V2\Sim3DR\readme.md"文件,按照说明执行编译,注意windows使用python,而不是python3;     
* 3.2 open "THREEDDFA_V2\FaceBoxes\utils\nms\readme.txt" and Follow the instructions inside to compile； when done copy "cpu_nms.cp311-win_amd64.pyd"(or cp312) to nms folder;  
* 打开"HREEDDFA_V2\FaceBoxes\utils\nms\readme.txt"文件,按照说明执行编译,注意编译好的文件cpu_nms.cp311-win_amd64.pyd 要从build里复制出来，现有的这个如果你是python312是无法使用的。目前的cpu_nms.cp311-win_amd64.pyd是torch2.51 cuda124，python311的 ；  

# 4. CHeckponit Required 
----
* 4.1 SHMT h4 or h0,download in [google driver](https://drive.google.com/drive/folders/1UJDdGCeE6qEqr3yi6BK1tEkwnjVKQDZY), 核心模型，h4或者h0 ，百度网盘，迟点上传
* 4.2 latent-diffusion [VQ-f4](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file)
* 4.3 face-parsing  huggingface [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing/tree/main)
* 4.4 3DDFA_V2：resnet22.pth [Google driver](https://drive.google.com/file/d/1dh7JZgkj1IaO4ZcSuBOBZl2suT9EPedV/view)  或者 [百度](https://pan.baidu.com/share/init?surl=IS7ncVxhw0f955ySg67Y4A) 提取码 lv1a
```
├── ComfyUI/models/SHMT
|         ├── epoch=000710-001.ckpt #h4   4.41G
|         ├── epoch=000755-001.ckpt #h0   4.41G
|         ├── model.ckpt  #latent-diffusion VQ-f4 721MB
|         ├── resnet22.pth  #3DDFA_V2   70.2MB

|── anypath/jonathandinu
|         ├── onnx
|             ├── model.onnx
|         ├── config.json
|         ├── model.safetensors
|         ├── preprocessor_config.json
|         ├── quantize_config.json
```


# Example
![](https://github.com/smthemex/ComfyUI_SHMT/blob/main/example_new.png)



# Citation
[@jonathandinu](https://github.com/jonathandinu) thanks her face-parsing

```
@article{sun2024shmt,
  title={SHMT: Self-supervised Hierarchical Makeup Transfer via Latent Diffusion Models},
  author={Sun, Zhaoyang and Xiong, Shengwu and Chen, Yaxiong and Du, Fei and Chen, Weihua, and Wang, Fang and Rong, Yi}
  journal={Advances in neural information processing systems},
  year={2024}
}
```

```
@inproceedings{guo2020towards,
    title =        {Towards Fast, Accurate and Stable 3D Dense Face Alignment},
    author =       {Guo, Jianzhu and Zhu, Xiangyu and Yang, Yang and Yang, Fan and Lei, Zhen and Li, Stan Z},
    booktitle =    {Proceedings of the European Conference on Computer Vision (ECCV)},
    year =         {2020}
}

@misc{3ddfa_cleardusk,
    author =       {Guo, Jianzhu and Zhu, Xiangyu and Lei, Zhen},
    title =        {3DDFA},
    howpublished = {\url{https://github.com/cleardusk/3DDFA}},
    year =         {2018}
}
```
```
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

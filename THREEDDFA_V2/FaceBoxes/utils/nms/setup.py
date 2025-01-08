# Enable cython support for eval scripts
# Run as
# setup.py build_ext --inplace
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
#################导入编译addToConfusionMatrix.pyx所需的依赖库#############
import os
from setuptools import setup, find_packages

has_cython = True

try:
    from Cython.Build import cythonize
except:
    print("Unable to find dependency cython. Please install for great speed improvements when evaluating.")
    print("sudo pip install cython")
    has_cython = False

include_dirs = []
try:
    import numpy as np
    
    include_dirs = np.get_include()
except:
    print("Unable to find numpy, please install.")
    print("sudo pip install numpy")

#################导入编译addToConfusionMatrix.pyx所需的依赖库#############


###设置C/C++编译器，可以用VS，这里本人机器已经安装Cygwin，所以支持g++编译器

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

#################设置C/C++编译器#########################################


#########################要编译的pyx文件路径#############################

pyxFile = os.path.join(cur_path, "cpu_nms.pyx")

# pyxFile = os.path.join("cityscapesscripts", "utils", "cython_bbox.pyx", #"cython_nms.pyx")
# pyxFile = os.path.join("cityscapesscripts", "utils", "cython_nms.pyx")

#########################要编译的pyx文件路径#############################


#########################采用cythonize进行编译#############################

ext_modules = []
if has_cython:
    ext_modules = cythonize(pyxFile)

#########################采用cythonize进行编译#############################


###########################设置编译参数###################################

config = {
    'name': 'it is me',  # optional参数
    'auyhor': 'it is me',  # optional参数
    'ext_modules': ext_modules,  # required参数
    'include_dirs': [include_dirs]  # required参数
}

###########################设置编译参数###################################

setup(**config)
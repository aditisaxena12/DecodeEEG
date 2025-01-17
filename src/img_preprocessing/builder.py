import torch.nn as nn
import torch.nn.parallel as parallel
import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Ensure the path to resnet.py is correct
resnet_path = Path(__file__).resolve().parent
sys.path.append(str(resnet_path))

import resnet
import vgg

import resnet
import vgg

def BuildAutoEncoder(model_name):

    # if args.arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
    #     configs = vgg.get_configs(args.arch)
    #     model = vgg.VGGAutoEncoder(configs)

    # elif args.arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
    #     configs, bottleneck = resnet.get_configs(args.arch)
    #     model = resnet.ResNetAutoEncoder(configs, bottleneck)
    
    # else:
    configs, bottleneck = resnet.get_configs(model_name)
    model = resnet.ResNetAutoEncoder(configs, bottleneck) 
    

    model = nn.DataParallel(model).cuda()

    return model
from PIL import Image

import torch
from torchvision.transforms import transforms

import sys
sys.path.append("./")

import utils
import img_preprocessing.builder as builder

import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Encoder():
    def __init__(self):
        self.model_path = "/home/aditis/decodingEEG/DecodeEEG/data/results/caltech256-resnet18.pth"
        print('=> torch version : {}'.format(torch.__version__))
        utils.init_seeds(1, cuda_deterministic=False)
        print('=> modeling the network ...')
        self.model = builder.BuildAutoEncoder("resnet18")     
        total_params = sum(p.numel() for p in self.model.parameters())
        print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

        print('=> loading pth from {} ...'.format(self.model_path))
        utils.load_dict(self.model_path, self.model)
 
        self.trans = transforms.Compose([
                    transforms.Resize(256),                   
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                  ])

        self.model.eval()
        print('=> completed decoder initilization ...')

        
    def encode(self, input):
        img = Image.open(input).convert("RGB")

        img = self.trans(img).unsqueeze(0).cuda()

        with torch.no_grad():

            code = self.model.module.encoder(img).cpu().numpy()
            print(code.shape)

        return code


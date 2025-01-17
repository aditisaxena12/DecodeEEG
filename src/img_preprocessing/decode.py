
import matplotlib.pyplot as plt

import torch

from torchvision.transforms import transforms

import sys
sys.path.append("./")

import utils
import img_preprocessing.builder as builder
from datetime import datetime

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Decoder():
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

        self.trans = transforms.ToPILImage()

        self.model.eval()
        print('=> completed decoder initilization ...')

        
    def decode(self, input):
        input_tensor = torch.from_numpy(input).float()
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        with torch.no_grad():

            output = self.model.module.decoder(input_tensor.cuda())
            output = self.trans(output.squeeze().cpu())
            plt.imshow(output)
            plt.savefig(f'/home/aditis/decodingEEG/DecodeEEG/src/visualize/figs/generation_{current_datetime}.jpg')

        return output

    



from PIL import Image
import numpy
import argparse
import os

import torch
from torchvision.transforms import transforms

import sys
sys.path.append("./")

import utils
import img_preprocessing.builder as builder
import encode

# from img_to_vec import Img2Vec

def encode(model, img):

    with torch.no_grad():

        code = model.module.encoder(img).cpu().numpy()

    return code


def extract_features(input_folder_path, output_folder_path, model):
    # clear output folder
    files= os.listdir (output_folder_path)
    for file in files:
        os.remove(output_folder_path + '/' + file)

    #iterate through all files - assuming images
    filenames= os.listdir (input_folder_path)
    #img2vec = Img2Vec()

    for filename in filenames:
        img = Image.open(input_folder_path+'/'+filename).convert('RGB')
        trans = transforms.Compose([
                    transforms.Resize(256),                   
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                  ])
        #vec = img2vec.get_vec(img)
        img = trans(img).unsqueeze(0).cuda()

        model.eval()

        code = encode(model, img)

        print(code.shape)

        outfile = "/"+filename.split('.')[0]
        numpy.save(output_folder_path+outfile,code)
        

def get_concept(images_folder, vectors_folder, model):
    #iterate over all concept folders
    filenames= os.listdir (images_folder)

    for foldername in filenames:
        input_folder  = images_folder + foldername
        output_folder = vectors_folder + foldername
        print("Processing ...",foldername)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        extract_features(input_folder,output_folder, model)
        
    

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', default='resnet18', type=str, 
                        help='backbone architechture')
    parser.add_argument('--resume', default='/home/aditis/decodingEEG/imagenet-autoencoder/results/data_list-resnet18/099.pth', type=str)  
    
    # Adding optional argument
    parser.add_argument("--train", help = "If True, process training images")
    parser.add_argument("--test", help = "If True, process test images")          
    
    args = parser.parse_args()

    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    print('=> torch version : {}'.format(torch.__version__))

    utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)


    if (args.train == "True" or args.train == "true"):
        print("Extracting features from Training Images (16540)")
        path_to_images = "/home/aditis/decodingEEG/DecodeEEG/data/images/training_images/"
        output_folder = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors/training/"
        get_concept(path_to_images, output_folder, model)
        
    if (args.test == "True" or args.test == "true"):
        print("Extracting features from Test Images (200)")
        path_to_images = "/home/aditis/decodingEEG/DecodeEEG/data/images/test_images/"
        output_folder = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors/test/"
        get_concept(path_to_images, output_folder, model)

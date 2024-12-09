from PIL import Image
import numpy
import argparse
import os

from img_to_vec import Img2Vec


def extract_features(input_folder_path, output_folder_path):
    # clear output folder
    files= os.listdir (output_folder_path)
    for file in files:
        os.remove(file)

    #iterate through all files - assuming images
    filenames= os.listdir (input_folder_path)
    img2vec = Img2Vec()

    for filename in filenames:
        img = Image.open(input_folder_path+'/'+filename).convert('RGB')
        vec = img2vec.get_vec(img)
        outfile = "/"+filename.split('.')[0]
        numpy.save(output_folder_path+outfile, vec)
        

def get_concept(images_folder, vectors_folder):
    #iterate over all concept folders
    filenames= os.listdir (images_folder)

    for foldername in filenames:
        input_folder  = images_folder + foldername
        output_folder = vectors_folder + foldername
        print("Processing ...",foldername)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        extract_features(input_folder,output_folder)
        
    

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("--train", help = "If True, process training images")
    parser.add_argument("--test", help = "If True, process test images")

    # Read arguments from command line
    args = parser.parse_args()

    if (args.train == "True" or args.train == "true"):
        print("Extracting features from Training Images (16540)")
        path_to_images = "/home/aditis/decodingEEG/DecodeEEG/data/images/training_images/"
        output_folder = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors/training/"
        get_concept(path_to_images, output_folder)
        
    if (args.test == "True" or args.test == "true"):
        print("Extracting features from Test Images (200)")
        path_to_images = "/home/aditis/decodingEEG/DecodeEEG/data/images/test_images/"
        output_folder = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors/test/"
        get_concept(path_to_images, output_folder)

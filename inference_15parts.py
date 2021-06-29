import os
import argparse
import sys
import pprint

parser = argparse.ArgumentParser(description='loading eval params')
parser.add_argument('--gpus', metavar='N', type=int, default=1)
parser.add_argument('--model', type=str, default='./weights/model_simulated_RGB_mgpu_scaling_append.0071.h5', help='path to the weights file')
parser.add_argument('--input_folder', type=str, default='./input', help='path to the folder with test images')
parser.add_argument('--output_folder', type=str, default='./output', help='path to the output folder')
parser.add_argument('--max', type=bool, default=True)
parser.add_argument('--average', type=bool, default=False)
parser.add_argument('--scale', action='append', help='<Required> Set flag', required=True)

args = parser.parse_args()

from pprint import pprint
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from keras.models import load_model
import code
import copy
import scipy.ndimage as sn
from PIL import Image
from tqdm import tqdm
from model_simulated_RGB101 import get_testing_model_resnet101
from human_seg.human_seg_gt import human_seg_combine_argmax
#=======================================================================================================================

import tensorflow as tf
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
#=======================================================================================================================

# for loading/processing the images
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import shutil
#=======================================================================================================================


right_part_idx = [2, 3, 4,  8,  9, 10, 14, 16]
left_part_idx =  [5, 6, 7, 11, 12, 13, 15, 17]
human_part = [0,1,2,4,3,6,5,8,7,10,9,12,11,14,13]
human_ori_part = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
seg_num = 15 # current model supports 15 parts only

def recover_flipping_output(oriImg, heatmap_ori_size, paf_ori_size, part_ori_size):

    heatmap_ori_size = heatmap_ori_size[:, ::-1, :]
    heatmap_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    heatmap_flip_size[:,:,left_part_idx] = heatmap_ori_size[:,:,right_part_idx]
    heatmap_flip_size[:,:,right_part_idx] = heatmap_ori_size[:,:,left_part_idx]
    heatmap_flip_size[:,:,0:2] = heatmap_ori_size[:,:,0:2]

    paf_ori_size = paf_ori_size[:, ::-1, :]
    paf_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    paf_flip_size[:,:,ori_paf_idx] = paf_ori_size[:,:,flip_paf_idx]
    paf_flip_size[:,:,x_paf_idx] = paf_flip_size[:,:,x_paf_idx]*-1

    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    part_flip_size[:,:,human_ori_part] = part_ori_size[:,:,human_part]
    return heatmap_flip_size, paf_flip_size, part_flip_size

def recover_flipping_output2(oriImg, part_ori_size):

    part_ori_size = part_ori_size[:, ::-1, :]
    part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))
    part_flip_size[:,:,human_ori_part] = part_ori_size[:,:,human_part]
    return part_flip_size

def part_thresholding(seg_argmax):
    background = 0.6
    head = 0.5
    torso = 0.8

    rightfoot = 0.55 
    leftfoot = 0.55
    leftthigh = 0.55
    rightthigh = 0.55
    leftshank = 0.55
    rightshank = 0.55
    rightupperarm = 0.55
    leftupperarm = 0.55
    rightforearm = 0.55
    leftforearm = 0.55
    lefthand = 0.55
    righthand = 0.55
    
    part_th = [background, head, torso, leftupperarm ,rightupperarm, leftforearm, rightforearm, lefthand, righthand, leftthigh, rightthigh, leftshank, rightshank, leftfoot, rightfoot]
    th_mask = np.zeros(seg_argmax.shape)
    for indx in range(15):
        part_prediction = (seg_argmax==indx)
        part_prediction = part_prediction*part_th[indx]
        th_mask += part_prediction

    return th_mask


def process (input_image, params, model_params):
    input_scale = 1.0

    oriImg = cv2.imread(input_image)
    flipImg = cv2.flip(oriImg, 1)
    oriImg = (oriImg / 256.0) - 0.5
    flipImg = (flipImg / 256.0) - 0.5
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    seg_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 15))

    segmap_scale1 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale2 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale3 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale4 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    segmap_scale5 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale6 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale7 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))
    segmap_scale8 = np.zeros((oriImg.shape[0], oriImg.shape[1], seg_num))

    for m in range(len(multiplier)):
        scale = multiplier[m]*input_scale
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = [ 0,
                0, 
                (imageToTest.shape[0] - model_params['stride']) % model_params['stride'],
                (imageToTest.shape[1] - model_params['stride']) % model_params['stride']
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))

        input_img = imageToTest_padded[np.newaxis, ...]
        
        print( "\tActual size fed into NN: ", input_img.shape)

        output_blobs = model.predict(input_img)

        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        if m==0:
            segmap_scale1 = seg
        elif m==1:
            segmap_scale2 = seg         
        elif m==2:
            segmap_scale3 = seg
        elif m==3:
            segmap_scale4 = seg


    # flipping
    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(flipImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = [ 0,
                0, 
                (imageToTest.shape[0] - model_params['stride']) % model_params['stride'],
                (imageToTest.shape[1] - model_params['stride']) % model_params['stride']
              ]
        
        imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0), (0, 0)))
        input_img = imageToTest_padded[np.newaxis, ...]
        print( "\tActual size fed into NN: ", input_img.shape)
        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        seg = np.squeeze(output_blobs[2])
        seg = cv2.resize(seg, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        seg_recover = recover_flipping_output2(oriImg, seg)

        if m==0:
            segmap_scale5 = seg_recover
        elif m==1:
            segmap_scale6 = seg_recover         
        elif m==2:
            segmap_scale7 = seg_recover
        elif m==3:
            segmap_scale8 = seg_recover

    segmap_a = np.maximum(segmap_scale1,segmap_scale2)
    segmap_b = np.maximum(segmap_scale4,segmap_scale3)
    segmap_c = np.maximum(segmap_scale5,segmap_scale6)
    segmap_d = np.maximum(segmap_scale7,segmap_scale8)
    seg_ori = np.maximum(segmap_a, segmap_b)
    seg_flip = np.maximum(segmap_c, segmap_d)
    seg_avg = np.maximum(seg_ori, seg_flip)

    
    return seg_avg

#╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
#. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : –
#. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : –
#██╗░░██╗███████╗██████╗░███████╗░░░░░░░░░██████╗████████╗░█████╗░██████╗░████████╗░██████╗░░░░░░░░███╗░░░███╗██╗░░░██╗
#██║░░██║██╔════╝██╔══██╗██╔════╝░░░░░░░░██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝░░░░░░░░████╗░████║╚██╗░██╔╝
#███████║█████╗░░██████╔╝█████╗░░░░░░░░░░╚█████╗░░░░██║░░░███████║██████╔╝░░░██║░░░╚█████╗░░░░░░░░░██╔████╔██║░╚████╔╝░
#██╔══██║██╔══╝░░██╔══██╗██╔══╝░░░░░░░░░░░╚═══██╗░░░██║░░░██╔══██║██╔══██╗░░░██║░░░░╚═══██╗░░░░░░░░██║╚██╔╝██║░░╚██╔╝░░
#██║░░██║███████╗██║░░██║███████╗░░░░░░░░██████╔╝░░░██║░░░██║░░██║██║░░██║░░░██║░░░██████╔╝░░░░░░░░██║░╚═╝░██║░░░██║░░░
#╚═╝░░╚═╝╚══════╝╚═╝░░╚═╝╚══════╝░░░░░░░░╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝░░░╚═╝░░░╚═════╝░░░░░░░░░╚═╝░░░░░╚═╝░░░╚═╝░░░
                                    #██████╗░██████╗░░█████╗░░██████╗░██████╗░░█████╗░███╗░░░███╗
                                    #██╔══██╗██╔══██╗██╔══██╗██╔════╝░██╔══██╗██╔══██╗████╗░████║
                                    #██████╔╝██████╔╝██║░░██║██║░░██╗░██████╔╝███████║██╔████╔██║
                                    #██╔═══╝░██╔══██╗██║░░██║██║░░╚██╗██╔══██╗██╔══██║██║╚██╔╝██║
                                    #██║░░░░░██║░░██║╚█████╔╝╚██████╔╝██║░░██║██║░░██║██║░╚═╝░██║
                                    #╚═╝░░░░░╚═╝░░╚═╝░╚════╝░░╚═════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝
#. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : –
#. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : – ⭒ ⊹ ⭒ – : ✧ : -˚̣⋅ .. ⋅ ˚̣- : ✧ : –
#╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
def k_means_clustering (answer):
    print("START Next Gen FEATURE EXTRACTION")

    import glob
    import shutil

    # Remove all files inside the feature directory
    files = glob.glob('/CDCL/features/*')
    for f in files:
        shutil.rmtree(f)


    # this list holds all the image filename
    pictures = []
    # In data are the key value pares "Name of file": "features"
    data = {}

    # The model that creates the feature vectors is loaded here
    model = VGG16(weights='imagenet')
    # model = VGG16(weights='imagenet', include_top=False)
    # model.summary()
    # # load model
    # model = VGG16()

    # Here the output layer gets removed
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    # Setting Input and output directory for feature vector generation
    directory = '/CDCL/output/'
    output_directory = '/CDCL/features/'

    # Reading images in folder named "output" (that acts now as our input folder) and safe names in pictures
    for filename in os.listdir(directory):
        pictures.append(filename)


    pictures = sorted(pictures)

    # In this section, a lot is taken over from: https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34
    for image_name in pictures:
        # load the image as a 224x224 array
        img = load_img(directory + image_name, target_size=(224,224))
        # convert from 'PIL.Image.Image' to numpy array
        img = np.array(img)

        reshaped_img = img.reshape(1,224,224,3)
        processed_image = preprocess_input(reshaped_img.astype(np.float64))

        # Here the neuronal network comes into action
        features = model.predict(processed_image)
        # And gets fed into the dict
        data[image_name] = features

    # Since we have now the feature vectors of all the images, a similarity search can be started.
    # Answer of the console gets analysed and in case of a yes, the similarity search gets started
    if answer == "yes" or answer == "y" or answer == "Y" or answer == "Yes":
        similarity_search(data, model)

    # get a list of the filenames
    filenames = sorted(np.array(list(data.keys())))

    # get a list of just the features
    feat = np.array(list(data.values()))

    # reshape so that there are 'numberofpictures' samples of 4096 vectors. Using it for PCA
    feat = feat.reshape(-1, 4096)

    #/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=
    # This part here would allow to add a PCA to reduce the dimentionality. For big datasets that is quite useful to
    # lower the calculation work
    #
    # Using PCA to reduce tumber of dimensions (from 4096 to 'numberofpictures')
    # pca = PCA(n_components=5, random_state=22)
    # pca.fit(feat)
    # x = pca.transform(feat)
    #/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=


    # Here the k-means clustering gets done. It will be calculated, no matter what the answer of the console is
    input_notok = True
    n_clusters = 10
    while input_notok:
        try:
            n_clusters = int(input("How many groups do you want to cluster? [e.g. 10]: "))
            input_notok = False
        except ValueError:
            print("Sorry, only enter integers > 0!")

    kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=22)
    kmeans.fit(feat)

    #print("K-Means labels are: ", kmeans.labels_)

    # This holds the cluster id and the images { id: [images] }
    groups = {}

    # The cluster groups get formed here
    for file, cluster in zip(filenames, kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    # /=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=
    #If you want to have the segmentet pictures in group directories
    # for groupkey in groups.keys():
    #     for picturename in groups[groupkey]:
    #         if os.path.isdir(output_directory + 'Group_' + str(groupkey)):
    #             shutil.copy2(directory + picturename, output_directory + 'Group_' + str(groupkey) + '/' + picturename)
    #         else:
    #             os.mkdir(output_directory + 'Group_' + str(groupkey))
    #             shutil.copy2(directory + picturename, output_directory + 'Group_' + str(groupkey) + '/' + picturename)
    # /=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=

    #If you want the non-segmentet pictures in group directories
    for groupkey in groups.keys():
        for picturename in groups[groupkey]:
            if os.path.isdir(output_directory + 'Group_' + str(groupkey)):
                if picturename == 'similarity_search_picture.jpg':
                    shutil.copy2(directory + picturename, output_directory + 'Group_' + str(groupkey) + '/' + picturename)
                else:
                    shutil.copy2('/CDCL/input/' + picturename[4:-4], output_directory + 'Group_' + str(groupkey) + '/' + picturename[4:-4])
            else:
                os.mkdir(output_directory + 'Group_' + str(groupkey))
                if picturename == 'similarity_search_picture.jpg':
                    shutil.copy2(directory + picturename, output_directory + 'Group_' + str(groupkey) + '/' + picturename)
                else:
                    shutil.copy2('/CDCL/input/' + picturename[4:-4], output_directory + 'Group_' + str(groupkey) + '/' + picturename[4:-4])

# =======================================================================================================================

def similarity_search (feature_vectors, model):
    import glob
    import shutil

    # Empty the output directory of the similarity search
    files = glob.glob('/CDCL/similarity_search_results/*')
    for f in files:
        os.remove(f)

    from scipy import spatial
    # This will contains as key the name of the picture and as value the distance to similarity picture
    similarity_map = {}

    # Since we need again to calculate the feature vector of the example image, this part of the code is pretty much
    # identical to the parts in the k-means function
    img = load_img('/CDCL/output/similarity_search_picture.jpg', target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)

    reshaped_img = img.reshape(1,224,224,3)
    processed_image = preprocess_input(reshaped_img.astype(np.float64))

    example_image_feature_vector = model.predict(processed_image)

    # The distance of the feature vectors between the example image and all the other feature vectors gets calculated
    # I use the cosine distance.
    for image_name in feature_vectors:
        distance = 1 - spatial.distance.cosine(feature_vectors[image_name], example_image_feature_vector)
        if image_name not in similarity_map.keys() and image_name != 'similarity_search_picture.jpg':
            similarity_map[image_name] = distance



    from collections import Counter

    c = Counter(similarity_map)

    # In most_common the names of the top x nearest images get saved
    input_notok = True
    n_similar = 10
    while input_notok:
        try:
            n_similar = int(input("How many similar images do you want to get? [e.g. 10]: "))
            input_notok = False
        except ValueError:
            print("Sorry, only enter integers > 0!")

    most_common = dict(c.most_common(n_similar))

    # Writing these images into their new directory. Parts of the name need to be trimmed since per default
    # CDCL adds a "seg_" at the beginning of the names and a ".jpg" at the end
    for groupkey in most_common.keys():
        if os.path.isdir('/CDCL/similarity_search_results'):
            shutil.copy2('/CDCL/input/' + groupkey[4:-4], '/CDCL/similarity_search_results/' + groupkey)
        else:
            os.mkdir('/CDCL/similarity_search_results')
            shutil.copy2('/CDCL/input/' + groupkey[4:-4], '/CDCL/similarity_search_results/' + groupkey)



# =======================================================================================================================

if __name__ == '__main__':

    args = parser.parse_args()
    keras_weights_file = args.model

    print('start processing...')
    # load model
    model = get_testing_model_resnet101() 
    model.load_weights(keras_weights_file)
    params, model_params = config_reader()

    scale_list = []
    for item in args.scale:
        scale_list.append(float(item))

    params['scale_search'] = scale_list

    # This will get the input of the console. It checks if the user wants to do a similarity search or not.
    answer = ''
    print("Do you want to make a similarity search to search the database for one specific gesture? yes/no")
    while True:
        answer = input()
        if answer == "yes" or answer == "y" or answer == "Y" or answer == "Yes":
            break
        elif answer == "no" or answer == "n" or answer == "N" or answer == "No":
            break
        else:
            print("I dont understand. Can you repeat? Please answer with yes/no")

    # »»————-This part only applies for when the similarity search is done. It calculates the segmentation of the————-««
    # »»————- example image and is mostly a copy of the part below. For documentation - see below————-««
    if answer == "yes" or answer == "y" or answer == "Y" or answer == "Yes":
        example_image_feature_vector = {}
        print("Please enter path to example image. A working test example would be: /CDCL/input/01christ.jpg")
        example_image = input()

        if example_image.endswith(".png") or example_image.endswith(".jpg"):
            print(args.input_folder + '/' + example_image)
            seg = process(example_image, params, model_params)
            seg_argmax = np.argmax(seg, axis=-1)
            seg_max = np.max(seg, axis=-1)
            th_mask = part_thresholding(seg_argmax)
            seg_max_thres = (seg_max > 0.1).astype(np.uint8)
            seg_argmax *= seg_max_thres
            seg_canvas = human_seg_combine_argmax(seg_argmax)

            seg_canvas[seg_canvas != 0] = 1
            seg_canvas[seg_canvas == 0] = 255
            seg_canvas[seg_canvas == 1] = 0

            cur_canvas = cv2.imread(example_image)

            canvas = cv2.subtract(cur_canvas, seg_canvas)

            filename = '%s/%s.jpg' % (args.output_folder, 'similarity_search_picture')#Intern hat similarity search bild immer diesen Namen
            cv2.imwrite(filename, canvas)
    # »»—————————————————————————————————————————————————————————————————————————————————————————————————————————————-««

    # generate image with body parts
    # »»————-I didnt change this part ☟ ————-««
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            print(args.input_folder+'/'+filename)
            seg = process(args.input_folder+'/'+filename, params, model_params)
            seg_argmax = np.argmax(seg, axis=-1)
            seg_max = np.max(seg, axis=-1)
            th_mask = part_thresholding(seg_argmax)
            seg_max_thres = (seg_max > 0.1).astype(np.uint8)
            seg_argmax *= seg_max_thres
    # »»————-I didnt change this part ☝︎ ————-««

            # The Masks are created in the human_seg_combine_argmax(). I did some alteration there to only have a
            # segmentation of the hands and arms.
            seg_canvas = human_seg_combine_argmax(seg_argmax)

            # Here the already created mask gets a colour change to black and white. That enables me to
            # devide the colour values of the original image by the mask. since the background is white,
            # it will be deleted. Since the hands and arms are black, nothing will change with a subtraction.
            seg_canvas[seg_canvas != 0] = 1
            seg_canvas[seg_canvas == 0] = 255
            seg_canvas[seg_canvas == 1] = 0

            cur_canvas = cv2.imread(args.input_folder+'/'+filename)

            # Here the subtraction is done. The background of the original image will be deleted
            canvas = cv2.subtract(cur_canvas, seg_canvas)

            filename = '%s/%s.jpg'%(args.output_folder,'seg_'+filename)
            cv2.imwrite(filename, canvas)


    k_means_clustering(answer)
    print("Program finished!!!!!!")




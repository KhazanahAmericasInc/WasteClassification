import os
import sys
import shutil, errno
import zipfile as zf
import random
from glob import glob
from pathlib import Path

from fastai.vision import *
from fastai.metrics import error_rate

from datetime import datetime
import h5py
import numpy as np
###########################################################################     
######################## data preprocessing ###############################
###########################################################################
'''
create folder: original_data; processed_data

get original data
unzip file
move data to the path of folder original_Data

copy original_data and paste to processed_data so we can process data in the right folder
split dataset into test and train subset

'''


############################### Helper functions ###############################

'''
create the folders
input: folders want to create in the format "./xxx/"
output: void
'''
def create_folders(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print("\nCreated", folder)
        else:
            inp = input('Do you clear the folder ' + folder + '?, y/n: ')
            if inp.lower() == "y":
                print("The folder will be cleared")
                try:
                    shutil.rmtree(folder)
                    os.makedirs(folder)
                except OSError as e:
                    print ("Error: %s - %s." % (e.filename, e.strerror))
            elif inp.lower() == "n":
                print("The folder will not be cleared")
            else:
                print("Please type y/n")
    return


'''
unzip the folder to the same path
input: zipfile to unzip
output: void
'''
def unzip(zipfile):
    files = zf.ZipFile(zipfile, 'r')
    files.extractall()
    files.close()
    
'''
move files from one folder to another
input: src folder and dst folder in format ./xxx/
output: void
'''
def move_folder(src, dst):
    try:
        files = get_subsets(src)
        for file in files:
            shutil.move(src+file, dst)
        if len(os.listdir(src)) == 0:
            os.rmdir(src)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

def rename_and_move_images(src, dst):
    try:
        files = get_subsets(src)
        for file in files:
            s = rename_image_name(file)
            os.rename(src + file, dst + s)
#             shutil.move(src+file, dst)
        os.rmdir(src)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

    
def rename_image_name(waste):
    if waste.startswith("plastic"):
        s = convert_name(waste, "recycle", "plastic") + "001" + ".jpg"
    elif waste.startswith("metal"):
        s = convert_name(waste, "recycle", "metal") + "002" + ".jpg"
    elif waste.startswith("cardboard"):
        s = convert_name(waste, "compost", "cardboard") + "001" + ".jpg"
    elif waste.startswith("paper"):
        s = convert_name(waste, "compost", "paper") + "002" + ".jpg"
#     elif waste.startswith("trash"):
#         s = waste[:-4] + "001" + ".jpg"
    elif waste.startswith("glass"):
        s = convert_name(waste, "recycle", "glass") + "003" + ".jpg"
    else:
        s = waste
    return s
  
def convert_name(src, dst, type):
  s = dst + src[len(type):-4]
  return s
                
'''
copy files from one folder to another
input: src folder and dst folder in format ./xxx/
output: void
'''
def copy_folder(src, dst):
    try: 
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    except OSError as e:
        if e.errno == errno.ENOTDIR:
            shutil.copy(Src, dst)
        else:
            raise

'''
get subsets (either files or folders) of the folder
input: folder path
output: list of name of subsets
'''
def get_subsets(path):
    subsets = os.listdir(path)
    for s in subsets:
        if s.startswith('.'):
            subsets.remove(s)
    return subsets

'''
get path of subsets (either files or folders) of the folder
input: folder path
output: list of subsets path
'''
def get_subsets_path(path):
    return glob(path+"*/")

#strip the name from a path
def get_name_from_path(f):
    return f[f.rindex("/")+1: ]

'''
split all data into train set and test set
input: parent path of trainset folder and testset folder, the ratio of test/all data
output: void
'''
def split_into_train_and_test_sets(datapath, ratio):
    assert ratio <= 1 and ratio >= 0
    test_path = os.path.join(datapath, "test/")
    train_path = os.path.join(datapath, "train/")
    print(test_path)
    subset_paths = get_subsets_path(train_path)
    subsets = get_subsets(train_path)
    print(subset_paths)
    print(subsets)
    for i, path in enumerate(subset_paths):
        curr = subsets[i]

        temp = test_path + curr + "/"
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        
        images = glob(path + "*.jpg")
        rand = random.sample(images, int(ratio*len(images)))
        print(curr , " -- size of test set: " , len(rand) , ", size of train set: " , (len(images)-len(rand)))

        for image in rand:
            dst = temp + get_name_from_path(image)
            os.rename(image, dst)

       
 # TODO
'''
Puts the image from an image path into the appropriate size, shape, and normalizes
the image
input: image_pth - the path to the image
        width - the width to resize the image to
        height - the height to resize the image to
        channels - [optional] the number of channels the image has, generally 3 (for RGB/BGR) or 1 for grayscale
'''
def process_single_img(image_pth, width, height, channels):
    #open image with open cv
    img = cv2.imread(image_pth)

    #convert to grayscale if 1 channel required for processing
    if(channels==1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    
    #resize and reshape the image
    img = cv2.resize(img, (width, height))
    img = img.reshape(width,height,channels)

    #normalize the image
    img = scale_X(img)
    return img
            
            
            
            
'''
export the data folder to a .h5 file
input: data folder path; eg. "./train_data"
output: void
'''
def export_to_h5(path):
    now = datetime.now()
    h5_file = "./exported_h5/waste_data" + now.strftime("%Y-%m-%d") + ".h5"
    f = h5py.File(h5_file, 'w')
    shape = (get_dataset_size(path+"**/*.jpg"), 64, 64, 3)

    train_set_x = f.create_dataset("train_set_x", shape, compression="gzip", compression_opts=9)
    
    label = []
    index = 0
    
    folder_paths = get_subsets_path(path)
    classes = get_subsets(path)
    # iterate through all the folders
    for i, folder in enumerate(folder_paths):
        images = glob(folder+"*.jpg")
        for j, img_pth in enumerate(images):
          img = process_single_img(img_pth, 64, 64)
          train_set_x[index] = img
          label.append(i)
          index += 1
          
    f.create_dataset("y", compression="gzip", compression_opts=9, data=label)
    
    f.close()
    
'''
get the size of dataset (number of images)
input: image path; eg. {image folder} + "*.jpg"
output: size
'''
def get_dataset_size(path):
    images = glob(path, recursive=True)
    img_num = len(images)
    return img_num


print("Importing data from local machine")
########## if compile locally ###############
datapath = "./processed_data/"
test_path = os.path.join(datapath, "test")
train_path = os.path.join(datapath, "train")
folders = ["./original_data/", datapath, test_path, train_path, "./trained_models/"]
create_folders(folders)
unzip("dataset-resized.zip")
move_folder("./dataset-resized/", "./original_data")


waste_types = ["compost", "recycle"]
waste_paths = ["./original_data/"+ w + "/" for w in waste_types]
create_folders(waste_paths)
rename_and_move_images("./original_data/plastic/","./original_data/recycle/")
rename_and_move_images("./original_data/metal/","./original_data/recycle/")
rename_and_move_images("./original_data/glass/","./original_data/recycle/")
rename_and_move_images("./original_data/cardboard/","./original_data/compost/")
rename_and_move_images("./original_data/paper/","./original_data/compost/")

copy_folder("./original_data/", train_path)
#         waste_types = get_subsets(datapath)
split_into_train_and_test_sets(datapath, 0.2)

#export_to_h5("./processed_data/train")








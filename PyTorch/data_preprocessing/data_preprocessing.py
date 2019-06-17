#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:57:08 2019

@author: admin
"""

#!/usr/bin/python
import os
import sys
import shutil, errno
import zipfile as zf
import random
from glob import glob
from pathlib import Path

from fastai.vision import *
from fastai.metrics import error_rate


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
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
        
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
split all data into train set, validation set and test set
input: parent path of trainset folder and testset folder, the ratio of test/all data
output: void
'''
def split_into_train_valid_and_test_sets(datapath, ratio1, ratio2):
    assert ratio1 <= 1 and ratio1 >= 0
    assert ratio2 <= 1 and ratio2 >= 0
    test_path = os.path.join(datapath, "test/")
    valid_path = os.path.join(datapath, "valid/")
    train_path = os.path.join(datapath, "train/")
    print(test_path)
    subset_paths = get_subsets_path(train_path)
    subsets = get_subsets(train_path)
    print(subset_paths)
    print(subsets)
    for i, path in enumerate(subset_paths):
        curr = subsets[i]

        temp = valid_path + curr + "/"
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        
        images = glob(path + "*.jpg")
        rand = random.sample(images, int(ratio1*len(images)))
        print(curr , " -- size of non-trainset: " , len(rand) , ", size of trainset: " , (len(images)-len(rand)))

        for image in rand:
            dst = temp + get_name_from_path(image)
            os.rename(image, dst)
    
    subset_paths = get_subsets_path(valid_path)
    subsets = get_subsets(valid_path)
    print(subset_paths)
    print(subsets)
    for i, path in enumerate(subset_paths):
        curr = subsets[i]

        temp = test_path + curr + "/"
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        
        images = glob(path + "*.jpg")
        rand = random.sample(images, int(ratio2*len(images)))
        print(curr , " -- size of testset: " , len(rand) , ", size of validset: " , (len(images)-len(rand)))

        for image in rand:
            dst = temp + get_name_from_path(image)
            os.rename(image, dst)
            
            
'''
split all data into train set, validation set and test set
input: parent path of trainset folder and testset folder, the ratio of test/all data
output: void
'''
def split_into_train_valid_and_test_sets(datapath, ratio1, ratio2):
    assert ratio1 <= 1 and ratio1 >= 0
    assert ratio2 <= 1 and ratio2 >= 0
    test_path = os.path.join(datapath, "test/")
    valid_path = os.path.join(datapath, "valid/")
    train_path = os.path.join(datapath, "train/")
    print(test_path)
    subset_paths = get_subsets_path(train_path)
    subsets = get_subsets(train_path)
    print(subset_paths)
    print(subsets)
    for i, path in enumerate(subset_paths):
        curr = subsets[i]

        temp = valid_path + curr + "/"
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        
        images = glob(path + "*.jpg")
        rand = random.sample(images, int(ratio1*len(images)))
        print(curr , " -- size of non-trainset: " , len(rand) , ", size of trainset: " , (len(images)-len(rand)))

        for image in rand:
            dst = temp + get_name_from_path(image)
            os.rename(image, dst)
    
    subset_paths = get_subsets_path(valid_path)
    subsets = get_subsets(valid_path)
    print(subset_paths)
    print(subsets)
    for i, path in enumerate(subset_paths):
        curr = subsets[i]

        temp = test_path + curr + "/"
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        
        images = glob(path + "*.jpg")
        rand = random.sample(images, int(ratio2*len(images)))
        print(curr , " -- size of testset: " , len(rand) , ", size of validset: " , (len(images)-len(rand)))

        for image in rand:
            dst = temp + get_name_from_path(image)
            os.rename(image, dst)
    
'''
split all data into train set and test set
input: parent path of trainset folder and testset folder, the ratio of test/all data
output: void
'''
def split_into_train_and_test_sets(datapath, ratio1, ratio2):
    assert ratio1 <= 1 and ratio1 >= 0
    assert ratio2 <= 1 and ratio2 >= 0
    test_path = os.path.join(datapath, "test/")
#    valid_path = os.path.join(datapath, "valid/")
    train_path = os.path.join(datapath, "train/")
    print(test_path)
    subset_paths = get_subsets_path(train_path)
    subsets = get_subsets(train_path)
    print(subset_paths)
    print(subsets)
    for i, path in enumerate(subset_paths):
        curr = subsets[i]

        temp = valid_path + curr + "/"
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        
        images = glob(path + "*.jpg")
        rand = random.sample(images, int(ratio1*len(images)))
        print(curr , " -- size of test set: " , len(rand) , ", size of trainset: " , (len(images)-len(rand)))

        for image in rand:
            dst = temp + get_name_from_path(image)
            os.rename(image, dst)

            
            
######################### main functions #########################          
        
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

datapath = "./processed_data/"
test_path = os.path.join(datapath, "test")
valid_path = os.path.join(datapath, "valid")
train_path = os.path.join(datapath, "train")
folders = ["./original_data/", datapath, test_path, train_path, valid_path, "./trained_models/"]
create_folders(folders)
unzip("dataset-resized.zip")
move_folder("./dataset-resized/", "./original_data")
copy_folder("./original_data/", train_path)
waste_types = get_subsets(datapath)
split_into_train_valid_and_test_sets(datapath, 0.4, 0.5)


path = Path(os.getcwd())/"processed_data"
tfms = get_transforms(do_flip=True, flip_vert=True)
data = ImageDataBunch.from_folder(path, test="test", ds_tfms=tfms, bs=16)


############################################################################     
############################# model training ###############################
############################################################################
#
#learn = create_cnn(data,models.resnet34,metrics=error_rate)
#
#learn.model
#
#learn.lr_find(start_lr=1e-6,end_lr=1e1)
#learn.recorder.plot()
#
## create directories for train and test sets
#for subset in subsets:
#    for waste_type in waste_types:
#        path = os.path.join(data_path, subset, waste_type)
#        if not os.path.exists(path):
#            os.makedirs(path)
            
            
            

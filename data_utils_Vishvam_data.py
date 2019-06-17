#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import shutil, errno
import zipfile as zf
import random
from glob import glob
from pathlib import Path

from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


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


# In[ ]:


'''
unzip the folder to the same path
input: zipfile to unzip
output: void
'''
def unzip(zipfile):
    files = zf.ZipFile(zipfile, 'r')
    files.extractall()
    files.close()
 


# In[ ]:


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
        


# In[ ]:


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


# In[ ]:


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
  


# In[ ]:


def convert_name(src, dst, type):
    s = dst + src[len(type):-4]
    return s
       


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


'''
split all data into train set and test set
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
    


# In[ ]:


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

        temp = valid_path + curr + "/"
        os.makedirs(os.path.dirname(temp), exist_ok=True)
        
        images = glob(path + "*.jpg")
        rand = random.sample(images, int(ratio*len(images)))
        print(curr , " -- size of test set: " , len(rand) , ", size of trainset: " , (len(images)-len(rand)))

        for image in rand:
            dst = temp + get_name_from_path(image)
            os.rename(image, dst)
            


# In[ ]:


'''
do the whole process of data processing with all the helper functions
input: using_colab: import data from different location; 
        if using_colab is 1, we are using colab
        else is 0, we are using local machine
        else, the user input is wrong, do nothing
output: data
'''
def process_data_pytorch(using_colab):

    if using_colab == 1:
        print("Importing data from google drive")
        ############## if compile on google colab #################
        # Load the Drive helper and mount
        from google.colab import drive

        # This will prompt for authorization.
        drive.mount('/content/drive')

        # !ls "/content/drive/My Drive"
        # os.chdir("../")
        datapath = "./processed_data/"
        test_path = os.path.join(datapath, "test")
        valid_path = os.path.join(datapath, "valid")
        train_path = os.path.join(datapath, "train")
        folders = ["./original_data/", datapath, test_path, train_path, valid_path, "./trained_models/"]
        du.create_folders(folders)
        du.unzip("/content/drive/My Drive/VishvamData.zip")
        du.move_folder("./VishvamData/", "./original_data")
        du.copy_folder("./original_data/", train_path)
    
        du.split_into_train_valid_and_test_sets(datapath, 0.4, 0.5)
    
    
        path = Path(os.getcwd())/"processed_data"
        tfms = get_transforms(do_flip=True, flip_vert=True)
        data = ImageDataBunch.from_folder(path, test="test", ds_tfms=tfms, bs=16)
#        datapath = "./processed_data/"
#        test_path = os.path.join(datapath, "test")
#        valid_path = os.path.join(datapath, "valid")
#        train_path = os.path.join(datapath, "train")
#        folders = ["./original_data/", datapath, test_path, train_path, valid_path, "./trained_models/"]
#        create_folders(folders)
#        unzip("/content/drive/My Drive/dataset-resized.zip")
#        move_folder("./dataset-resized/", "./original_data")
#        
#        
#        waste_types = ["compost", "recycle"]
#        waste_paths = ["./original_data/"+ w + "/" for w in waste_types]
#        create_folders(waste_paths)
#        rename_and_move_images("./original_data/plastic/","./original_data/recycle/")
#        rename_and_move_images("./original_data/metal/","./original_data/recycle/")
#        rename_and_move_images("./original_data/glass/","./original_data/recycle/")
#        rename_and_move_images("./original_data/cardboard/","./original_data/compost/")
#        rename_and_move_images("./original_data/paper/","./original_data/compost/")
#        
#        copy_folder("./original_data/", train_path)
#        
#        split_into_train_valid_and_test_sets(datapath, 0.4, 0.5)
#
#
#        path = Path(os.getcwd())/"processed_data"
#        tfms = get_transforms(do_flip=True, flip_vert=True)
#        data = ImageDataBunch.from_folder(path, test="test", ds_tfms=tfms, bs=16)

        return data
    
    elif using_colab == 0:
        print("Importing data from local machine")
        ########## if compile locally ###############
        datapath = "./processed_data/"
        test_path = os.path.join(datapath, "test")
        valid_path = os.path.join(datapath, "valid")
        train_path = os.path.join(datapath, "train")
        folders = ["./original_data/", datapath, test_path, train_path, valid_path, "./trained_models/"]
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
        
        split_into_train_valid_and_test_sets(datapath, 0.4, 0.5)


        path = Path(os.getcwd())/"processed_data"
        tfms = get_transforms(do_flip=True, flip_vert=True)
        data = ImageDataBunch.from_folder(path, test="test", ds_tfms=tfms, bs=16)

        return data


# In[ ]:


'''
do the whole process of data processing with all the helper functions
input: using_colab: import data from different location; 
        if using_colab is 1, we are using colab
        else is 0, we are using local machine
        else, the user input is wrong, do nothing
output: the path of the exported .h5 file
'''
def process_data_keras(using_colab):

    if using_colab == 1:
        print("Importing data from google drive")
        ############## if compile on google colab #################
        # Load the Drive helper and mount
        from google.colab import drive

        # This will prompt for authorization.
        drive.mount('/content/drive')

        # !ls "/content/drive/My Drive"
        # os.chdir("../")
        datapath = "./processed_data/"
        test_path = os.path.join(datapath, "test")
        train_path = os.path.join(datapath, "train")
        folders = ["./original_data/", datapath, test_path, train_path, "./trained_models/", "./exported_h5/"]
        create_folders(folders)
        unzip("/content/drive/My Drive/dataset-resized.zip")
        move_folder("./dataset-resized/", "./original_data")
#         copy_folder("./original_data/", train_path)
#         waste_types = get_subsets(datapath)

        waste_types = ["compost", "recycle"]
        waste_paths = ["./original_data/"+ w + "/" for w in waste_types]
        create_folders(waste_paths)
        rename_and_move_images("./original_data/plastic/","./original_data/recycle/")
        rename_and_move_images("./original_data/metal/","./original_data/recycle/")
        rename_and_move_images("./original_data/glass/","./original_data/recycle/")
        rename_and_move_images("./original_data/cardboard/","./original_data/compost/")
        rename_and_move_images("./original_data/paper/","./original_data/compost/")
        
        copy_folder("./original_data/", train_path)

        split_into_train_and_test_sets(datapath, 0.2)

        now = datetime.now()
        h5_file = "./exported_h5/waste_data" + now.strftime("%Y-%m-%d") + ".h5"

        return h5_file
    
    elif using_colab == 0:
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

        split_into_train_and_test_sets(datapath, 0.2)

        now = datetime.now()
        h5_file = "./exported_h5/waste_data" + now.strftime("%Y-%m-%d") + ".h5"

        return h5_file


# In[ ]:
def rename_images(src, wastetype):
    try:
        files = get_subsets(src)
        for i, file in enumerate(files):
            s = wastetype + str(i) + ".jpg"
            os.rename(src + file, src + s)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))




# In[ ]:

#waste_type = ["trash", "compost", "recycle"]
#for w in waste_type:
#    rename_images(("./data-merged/test/" + w + "/"), w)



# In[ ]:

#def rename_and_move_images(src, dst, wastetype):
#    try:
#        files = get_subsets(src)
#        for file in files:
#            os.rename(src + file, dst + file)
##             shutil.move(src+file, dst)
##        os.rmdir(src)
#    except OSError as e:
#        print ("Error: %s - %s." % (e.filename, e.strerror))
#        



# In[ ]:

def copytree(src, dst):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
            
#            
#waste_type = ["Trash", "Compost", "Recycling"]
#for w in waste_type:
#    copytree("../waste-classifier/clumped_kitchen_test_data_v3/" + w + "/", "../waste-classifier/original_data/" + w + "/")
#    
#    
    
    
    
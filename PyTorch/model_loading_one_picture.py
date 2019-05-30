#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:18:21 2019

@author: admin
"""

import os
import sys
import shutil, errno
import zipfile as zf
import random
from glob import glob
from pathlib import Path

from fastai.vision import *
from fastai.metrics import error_rate
from sklearn.metrics import confusion_matrix

import data_utils

'''
load dataset then load model then pass the dataset to the model for prediction
'''
data = data_utils.process_data_pytorch(0)

new = load_learner("./PyTorch/trained_models/")

from datetime import datetime
start = datetime.now()
img = data.test_ds[2][0]
new.predict(img)
end = datetime.now()

duration = (end - start).total_seconds() * 1000
print(duration)
print("millisecond per image")
# import glob
# file_size = len(glob.glob('./processed_data/test/*/*'))
# speed = duration/file_size



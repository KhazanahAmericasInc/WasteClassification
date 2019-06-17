#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:18:21 2019

@author: admin
"""

from fastai.vision import *
from fastai.metrics import error_rate

import data_utils

'''
load dataset then load model then pass the dataset to the model for prediction
'''
data = data_utils.process_data_pytorch(0)

new = load_learner("./PyTorch/trained_models/")

from datetime import datetime
start = datetime.now()
img = data.test_ds[2][0]
pred = new.predict(img)
print(pred)
end = datetime.now()

duration = (end - start).total_seconds() * 1000
print(duration)
print("millisecond per image")
# import glob
# file_size = len(glob.glob('./processed_data/test/*/*'))
# speed = duration/file_size



# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 22:27:36 2022

@author: xiaofan
"""

import os
import tarfile
import urllib 
import matplotlib.pyplot as plt
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handsonml2/master/" 
HOUSING_PATH = os.path.join("datasets", "housing") 
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"



housing.hist(bins=50, figsize=(20,15))
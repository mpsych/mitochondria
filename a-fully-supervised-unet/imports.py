#general
import os
import zipfile
import io
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

#ml and imaging
from scipy import misc
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from scipy.ndimage.filters import median_filter
import cv2
import mahotas as mh
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import backend as keras
import keras.callbacks
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import generator_extended as extension

#helper
import tqdm
from tqdm import tqdm
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')
MEMORY = 1
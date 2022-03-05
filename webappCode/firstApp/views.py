from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
# Create your views here.

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
import numpy as np
import pandas as pd 
import os
import zipfile
import tensorflow as tf
from tensorflow import keras
import nibabel as nib
from scipy import ndimage
from tensorflow import keras
#function to read the input nii for a given file
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan
#function to normalize the image
def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume
#function to resize the nii file converted to an array
def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
def process_scan(path):
    """Read and resize the volume to the desired  volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


model=load_model('./model/model87.h5')
    



def index(request):
    context={'a':1}
    return render(request, "index.html", context)

def predictImage(request):
    print(request)
    print (request.POST.dict())
    #print(request.FILES["filePath"])
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    print(testimage)
    #process_scan(testimage)
    #model.load_weights("3d_image_classification.h5")
    xnew=np.array(process_scan(str(testimage)))
    XnewX=np.expand_dims(xnew, axis=0)
    predictedLabel = model.predict(XnewX)[0]
    Pred_1= int((1 - predictedLabel[0])*100)
    Pred_2= int (predictedLabel[0]*100)
    #scores = [1 - predictedLabel[0], predictedLabel[0]]
    #class_names = ["normal", "abnormal"]
    #for score, name in zip(scores, class_names):
        
    #    print(
    #        "This model is %.2f percent confident that CT scan is %s"
    #        % ((100 * score), name)
    #    )

    
    context={'filePathName':filePathName,'scores1':Pred_1,'scores2':Pred_2}
    return render(request,'index.html',context)
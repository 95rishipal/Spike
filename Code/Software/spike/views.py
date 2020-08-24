from django.http import HttpResponse 
from django.shortcuts import render, redirect
from .Spike import Spike
import tensorflow as tf 
from .forms import *
import os, os.path
# tf.compat.v1.disable_eager_execution()
global spike
spike = Spike(256,256,150,6)
spike.Spike_Model('.\spike\Apple.h5')
my_dir = 'static/mysite/'
for fname in os.listdir(my_dir):
    if fname.startswith("sample_"):
        os.remove(os.path.join(my_dir, fname))


def index(request):
    layername = spike.Layer_Names()
    return render(request, 'index.html',{'data':'', 'layername':layername})
    
def thesis(request):
    layername = spike.Layer_Names()
    return render(request, 'thesis.html')

def features(request):
    form = UploadFileForm(request.POST, request.FILES)
    layername = spike.Layer_Names()
    # print(layername)
    return render(request, 'Model.html',{'form': form, 'layername':layername})

def classification(request):
    form = UploadFileForm(request.POST, request.FILES)
    layername = spike.Layer_Names()
    # print(layername)
    return render(request, 'classification.html')


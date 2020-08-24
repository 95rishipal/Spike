import json
from django.http import HttpResponse, Http404 
from .forms import *
from .Spike import Spike
from django.http import JsonResponse
import time
from PIL import Image
import io
import base64
import serial
import time 

global spike
spike = Spike(256,256,150,6)
modelflag = "Best"

def connect():
    ArduinoSerial = serial.Serial('com3',9600)
    time.sleep(4)
    return ArduinoSerial

def loadTestData(request):
    # print("Ajax Call Load Test Images")
    if request.is_ajax() and request.POST:
        # print("Read Images")
        name = request.POST.get('fruit')
        spike.Spike_Model('.\spike\{}{}.h5'.format(modelflag,name))
        spike.Compile_Model()
        xname = 'test_X_{}.npy'.format(name)
        yname = 'test_Y_{}.npy'.format(name)
        total = spike.Load_Test_Dataset(xname,yname,1)
        data = {'result': str(total)}  
        # print(data)
        return JsonResponse(data)
    else:
        return HttpResponse('Not Found')
        
def loaddata(request):
    # print("Ajax Call Load Test Images")
    if request.is_ajax() and request.POST:
        # print("Load Images")
        name = request.POST.get('fruit')
        spike.Spike_Model('.\spike\{}{}.h5'.format(modelflag,name))
        spike.Compile_Model()
        xname = 'test_X_{}.npy'.format(name)
        yname = 'test_Y_{}.npy'.format(name)
        per = int(request.POST.get('per'))
        per = per/100
        total = spike.Load_Test_Dataset(xname,yname,per)  
        # print(per)
        data = {'msg': "Loaded"} 
        return JsonResponse(data)

    else:
        return HttpResponse('Not Found')

def evaluate(request):
    # print("Ajax Call Evaluate")
    if request.is_ajax() and request.POST:
        # print("Evluation Starts")
        # print(spike.x_test.shape)
        result  = spike.Eval()
        data = {'msg': "Evaluation Done", 'loss': str(result[0]), 'acc': str(result[1])} 
        # print("Evluation Ends")
        return JsonResponse(data)

    else:
        return HttpResponse('Not Found')

def history(request):
    # print("Ajax Call History")
    if request.is_ajax() and request.POST: 
        # print("History Start")
        data = spike.history
        return JsonResponse(data)
    else:
        return HttpResponse('Not Found')

def upload_img(request):
    # print("Ajax Call upload image")
    if request.is_ajax() and request.POST:
        # print("Read Image")
        name = request.POST.get('fruit')
        filename = request.POST.get('name')
        form = UploadFileForm(request.POST, request.FILES)
        handle_uploaded_file(request.FILES['file'],filename)
        data = {'message': "Ajax Is Working !!!"}
        time.sleep(2)
        
        # print(name)
        spike.Spike_Model('.\spike\{}{}.h5'.format(modelflag,name))
        spike.Compile_Model()
        filename = 'static\mysite\sample_'+filename+'.png'
        result = spike.Pred(filename)
        input1 = result[0]
        try: 
            ArduinoSerial = connect()
            ArduinoSerial.write((str(input1)+"\n").encode())
        except:
            print("Arduino Is Not Connected")
        data = {'result': str(result),'name':filename}  
        # print(data)
        return JsonResponse(data)
    else:
        return HttpResponse('Not Found')

def upload_img_capture(request):
    # print("Ajax Call upload image")
    if request.is_ajax() and request.POST:
        # print("Read Image")
        name = request.POST.get('fruit')
        filename = request.POST.get('name')
        img = request.POST.get('file')
        # print(img)
        z = img[img.find('/9'):]
        im = Image.open(io.BytesIO(base64.b64decode(z))).save('./static/mysite/sample_'+filename+'.png')
        data = {'message': "Ajax Is Working !!!"}
        time.sleep(2)
        # print(name)
        spike.Spike_Model('.\spike\{}{}.h5'.format(modelflag,name))
        spike.Compile_Model()
        filename = 'static\mysite\sample_'+filename+'.png'
        result = spike.Pred(filename)
        try:
            input1 = result[0] 
            ArduinoSerial = connect()
            ArduinoSerial.write((str(input1)+"\n").encode())
        except:
            print("Arduino Is Not Connected")
        data = {'result': str(result),'name':filename}  
        # print(data)
        return JsonResponse(data)
    else:
        return HttpResponse('Not Found')



def handle_uploaded_file(f,name):
    # print(f)
    with open('./static/mysite/sample_'+name+'.png', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def features(request):
    # print("Ajax Call features")
    if request.is_ajax() and request.POST:
        layer = int(request.POST.get('layer'))
        name = request.POST.get('name')
        print(name)
        # print("Layer: "+ str(layer))
        # spike.Compile_Model()
        uri = spike.Layer_Feature(name,layer)
        return JsonResponse({'data':uri, 'layer':layer})
    else:
        return HttpResponse('Not Found')
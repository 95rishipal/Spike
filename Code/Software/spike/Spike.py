from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input, Concatenate,Add, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt 
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
from sklearn.datasets import load_files
from tensorflow.keras.utils import to_categorical
import base64
import urllib
import io
import math
from sklearn.svm import SVC
import numpy as np
from numpy import asarray
from numpy import save, load
import datetime
g1=tf.Graph()
g2=tf.Graph()
sess1=tf.Session(graph=g1)
sess2=tf.Session(graph=g2)
class Spike():
    class MyCustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, outer_instance):
            self.outer_instance = outer_instance

        def on_test_batch_end(self, batch,logs=None):
            n = { 'loss':str(logs['loss']),'acc': str(logs['acc'])}
            self.outer_instance.history[batch] = n
            print('For batch {}, loss is {:7.2f}, acc is {}'.format(batch, logs['loss'],logs['acc']))

    def __init__(self, width,height, batch_size, classes):
        self.img_height = height
        self.img_width = width
        self.n_batch = batch_size
        self.no_of_classes = classes
        self.history={}
    
    def convert_image_to_array(files):
        images_as_array=[]
        for file in files:
            # Convert to Numpy Array
            images_as_array.append(img_to_array(load_img(file,target_size=(256,256,3))))
        return images_as_array

    def Spike_Model(self,path):
        global g1,sess1
        with g1.as_default():
           with sess1.as_default():
            # load model
            self.model = load_model(path)
            self.model.summary()
            self.graph = tf.get_default_graph()

    def load_dataset(self,path):
        data = load_files(path)
        files = np.array(data['filenames'])
        targets = np.array(data['target'])
        target_labels = np.array(data['target_names'])
        return files,targets,target_labels

    def convert_image_to_array(self,files):
        images_as_array=[]
        for file in files:
            # Convert to Numpy Array
            images_as_array.append(img_to_array(load_img(file,target_size=(256,256,3))))
        return images_as_array

    def create_save(self):
        x_test, y_test,target_labels = self.load_dataset(path)
        sel = int(x_test.shape[0]*per)
        x_test = x_test[:sel]
        y_test = y_test[:sel]
        no_of_classes = len(np.unique(y_test))
        y_test = to_categorical(y_test,no_of_classes)
        x_test = np.array(self.convert_image_to_array(x_test))
        x_test = x_test.astype('float32')/255
        save('test_X_Apple.npy', x_test)
        save('test_y_Apple.npy', y_test)    
    

    def Eval(self):
        global g1,sess1
        with g1.as_default():
           with sess1.as_default():
            # load model
            print('Model Evaluate')
            self.history={}
            # callbacks = [self.history]
            return self.model.evaluate(self.x_test,self.y_test,callbacks = [self.MyCustomCallback(self)])
    
    def Load_Test_Dataset(self,pathX,pathY,per):
        global g1,sess1
        with g1.as_default():
           with sess1.as_default():
            self.x_test = load(pathX)
            self.y_test = load(pathY)
            size = int(self.x_test.shape[0]*per)
            self.x_test = self.x_test[:size]
            self.y_test = self.y_test[:size]
            # print("Dataset Loaded: "+str(self.x_test.shape[0]))
            # print(self.x_test.shape)
            # print(self.y_test.shape)
            return self.x_test.shape[0]

    def Compile_Model(self):
        global g1,sess1
        with g1.as_default():
           with sess1.as_default():
                adam = optimizers.Adam(learning_rate=0.001)
                self.model.compile(loss='categorical_crossentropy',
                            optimizer=adam,
                            metrics=['accuracy'])
                # print('Compiled!')

    def Layer_Feature(self,imgname,layer):
        global g1,sess1
        with g1.as_default():
           with sess1.as_default():
                img = img_to_array(load_img(imgname,target_size=(self.img_width,self.img_height,3)))
                getFeature = K.function([self.model.layers[0].input, K.learning_phase()],
                                        [self.model.layers[layer].output])
                # session = tf.compat.v1.keras.backend.get_session()
                # init =  tf.compat.v1.global_variables_initializer()
                # session.run(init)
                
                img = tf.keras.backend.expand_dims(img, axis=0)
                # print(img.shape)
                exTrain = getFeature([img, 0])
                filters = exTrain[0]
                f_min, f_max = filters.min(), filters.max()
                filters = (filters - f_min) / (f_max - f_min)
                lshape = self.model.layers[layer].output_shape
                if lshape[-1] == 16:
                    square = 4
                elif lshape[-1] == 32:
                    square = 5  
                else:
                    square = 8    
                ix = 1
                fig = plt.figure(figsize=(10,10),dpi= 50)
                # print(filters.shape)
                for _ in range(square):
                    for _ in range(square):
                        ax = plt.subplot(square, square, ix)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        plt.imshow(filters[0, :, :, ix-1], cmap='gray')
                        ix += 1
                # plt.show()
                buf = io.BytesIO()
                fig.savefig(buf,format = 'png',transparent=True, bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                string = base64.b64encode(buf.read())
                uri = urllib.parse.quote(string)
                return uri

    def Layer_Names(self):
        global g1,sess1
        with g1.as_default():
           with sess1.as_default():
                i=0
                layername ={}
                for layer in self.model.layers:
                    i +=1
                    if(layer.name.__contains__('conv')|layer.name.__contains__('max')|layer.name.__contains__('act')):
                        layername[i-1]=(layer.name,layer.output_shape)
                        # print(layer.name, i, layer.get_weights()[0].shape)    
                    # print(layername, layer.name, i)
        return layername

    def Pred(self,imgname):
        global g1,sess1
        with g1.as_default():
           with sess1.as_default():
                img = img_to_array(load_img(imgname,target_size=(self.img_width,self.img_height,3)))
                img = np.expand_dims(img,0)
                getFeature = K.function([self.model.layers[0].input, K.learning_phase()],
                                        [self.model.layers[-1].output])
                # print(img.shape)
                exTrain = getFeature([img, 0])
                # print(exTrain)
                return np.argmax(exTrain[0],axis=1)
    
    def ss(self,imgname):
        global g1,sess1
        with g1.as_default():
           with sess1.as_default():
                img = img_to_array(load_img(imgname,target_size=(self.img_width,self.img_height,3)))
                img = np.expand_dims(img,0)
                # print(img.shape)
                y = np.asarray([1])
                # print(y.shape)
                exTrain = self.model.evaluate(img,y,verbose=1)
                print(exTrain)
                return np.argmax(exTrain,axis=1)

if __name__ == '__main__':
    # print("Spike Model Loaded")
    # spike = Spike(256,256,150,6)
    # spike.Spike_Model('.\spike\Apple.h5')
    # spike.Compile_Model()
    # spike.Layer_Names()
    # uri = spike.Layer_Feature('static\mysite\sample.png',15)
    # print(spike.Pred('static\mysite\sample.png'))
    spike.Load_Test_Dataset('test_X_Apple.npy','test_Y_Apple.npy',0.1)
    print(spike.Eval())
    
    

    

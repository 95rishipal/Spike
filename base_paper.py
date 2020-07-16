"""
@ Author Rishipal Singh, Rajneesh Rani, Aman Kamboj
NIT Jalandhar
"""
# pip install split-folders tqdm
import split_folders
import tensorflow as tf
from sklearn.datasets import load_files
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras import Input
from numpy import argmax
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import seaborn as sn

split_folders.ratio("Source Folder", output="Destination Folder",seed=1337, ratio=(.5,.4,.1))


train_dir =  #train directory
test_dir =  #test directory
val_dir =  #validation directory
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels

print('Loading Trainning Data:')
x_train, y_train,target_labels = load_dataset(train_dir)
print('Loading Testing Data:')
x_test, y_test,_ = load_dataset(test_dir)
print('Loading Validation Data:')
x_val, y_val,_ = load_dataset(val_dir)
print('Loading complete!')

print('Training set size : ' , x_train.shape)
print('Testing set size : ', x_test.shape)
print('Validation set size: ',x_val.shape)

no_of_classes = len(np.unique(y_train))


y_train = to_categorical(y_train,no_of_classes)
y_test = to_categorical(y_test,no_of_classes)
y_val = to_categorical(y_val,no_of_classes)
y_train[0] # Note that only one element has value 1(corresponding to its label) and others are 0.
y_train.shape
print(x_test.shape,y_test.shape)
print(x_train.shape,y_train.shape)
print(x_val.shape,y_val.shape)

def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        images_as_array.append(img_to_array(load_img(file,target_size=(256,256,3))))
    return images_as_array

x_train = np.array(convert_image_to_array(x_train))
print('Training set shape : ',x_train.shape)

x_val = np.array(convert_image_to_array(x_val))
print('Validation set shape : ',x_val.shape)

x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ',x_test.shape)

x_train = x_train.astype('float32')/255
x_val = x_val.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train[0]

fig = plt.figure(figsize =(9,3))
for i in range(10):
    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
plt.show()

"""## Fine tune VGG16 Model"""
baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(256, 256, 3)))
headModel = baseModel.output
headModel = Flatten(name = "flattern")(headModel)
headModel = Dense(512, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)


#Define Optimizers SGD, ADAM, RMS
sgd = optimizers.SGD(lr=0.001, momentum=0.9)
rms = optimizers.RMSprop(0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])
print('Compiled!')

batch_size = 40
checkpointer = ModelCheckpoint(filepath = 'Kiwi.hdf5', monitor='val_acc', mode='max', verbose = 1, save_best_only = True)
early_stop = EarlyStopping(monitor='val_loss', patience=30)
history = model.fit(x_train,y_train,
        batch_size = batch_size,
        epochs=40,
        validation_data=(x_val, y_val),
        callbacks = [checkpointer, early_stop], 
        shuffle=True)


test_pred_raw = model.predict(test_images)
test_pred = np.argmax(test_pred_raw, axis=1)

# Calculate the confusion matrix.
cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)

# Log the confusion matrix as an image summary.
figure = plot_confusion_matrix(cm, class_names= ['Kiwi A', 'Kiwi B', 'Kiwi C'])
predicted=model.predict(x_test)
actual = argmax(y_test,axis=1)
results = confusion_matrix(actual, predicted) 
print(results) 
print ('Accuracy Score :',accuracy_score(actual, predicted)) 
print ('Report : ')
print (classification_report(actual, predicted)) 
df_cm = pd.DataFrame(results)
plt.figure(figsize = (5,5))
sn.heatmap(df_cm, annot=False, center=0, cbar_kws={'label': 'My Colorbar', 'orientation': 'horizontal'})

score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

#Finally lets visualize the loss and accuracy wrt epochs
plt.figure(figsize=(16, 16)) 
   
# summarize history for accuracy  
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
   
# summarize history for loss  
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()


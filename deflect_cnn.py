# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 18:38:30 2021

@author: Юрий
"""


import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import listdir, mkdir
from os.path import isfile, join
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


path_to_image_folder = 'K:\\Work\\Python\\GitHub\\Deflect_CNN\\2_parameters_data\\'
path_to_output = 'K:\\Work\\Python\\GitHub\\Deflect_CNN\\output\\'
try:
    mkdir(path_to_output+"checkpoint")
    mkdir(path_to_output+"saved")
except:
    pass 
   
dim = (256,256)
save_model = True
use_saved_model = False
shifts=True
model_name = 'proton_deflect_demo_with_shifts'


onlyfiles = [f for f in listdir(path_to_image_folder) if isfile(join(path_to_image_folder, f))]
items=len(onlyfiles)


k=0
max_val_num=0
max_train_num=0
max_test_num=0
for i in onlyfiles:
    val_num=0
    if i[:7]=='str_val':
        max_val_num+=1
    if i[:7]=='str_tra':
        max_train_num+=1
    if i[:7]=='str_tes':
        max_test_num+=1

x0_test = np.empty((max_test_num, dim[1], dim[0], 1))
y0_test = np.empty((max_test_num,2))

x0_val = np.empty((max_val_num, dim[1], dim[0], 1))
y0_val = np.empty((max_val_num,2))

x0_train = np.empty((max_train_num, dim[1], dim[0], 1))
y0_train = np.empty((max_train_num,2))

    


k=0
val_num=0
train_num=0
test_num=0
for i in onlyfiles:
    if i[:7]=='str_val':
        image = Image.open(path_to_image_folder+i).convert('LA')
        x0_val[val_num,:,:,0] = np.array( image.resize(dim))[:,:,0]
        for k in range(len(i)):
            if i[k]=='I' and i[k+1]=='=':
                I_start=k+2
            if i[k]=='A':
                I_finishi=k
            if i[k]=='U' and i[k+1]=='=':
                U_start=k+2
            if i[k]=='k':
                U_finish=k
        y0_val[val_num,0] = float(i[I_start:I_finishi])
        y0_val[val_num,1] = float(i[U_start:U_finish])
        val_num+=1
        
    if i[:7]=='str_tes':
        image = Image.open(path_to_image_folder+i).convert('LA')
        x0_test[test_num,:,:,0] = np.array( image.resize(dim))[:,:,0]
        for k in range(len(i)):
            if i[k]=='I' and i[k+1]=='=':
                I_start=k+2
            if i[k]=='A':
                I_finishi=k
            if i[k]=='U' and i[k+1]=='=':
                U_start=k+2
            if i[k]=='k':
                U_finish=k
        y0_test[test_num,0] = float(i[I_start:I_finishi])
        y0_test[test_num,1] = float(i[U_start:U_finish])
        test_num+=1
        
    if i[:7]=='str_tra':
        image = Image.open(path_to_image_folder+i).convert('LA')
        x0_train[train_num,:,:,0] = np.array( image.resize(dim))[:,:,0]
        for k in range(len(i)):
            if i[k]=='I' and i[k+1]=='=':
                I_start=k+2
            if i[k]=='A':
                I_finishi=k
            if i[k]=='U' and i[k+1]=='=':
                U_start=k+2
            if i[k]=='k':
                U_finish=k
        y0_train[train_num,0] = float(i[I_start:I_finishi])
        y0_train[train_num,1] = float(i[U_start:U_finish])
        train_num+=1
        

    
x_train_min = np.min(x0_train)
x_train_max = np.max(x0_train)
y_train_min_I = np.min(y0_train[:,0])
y_train_max_I = np.max(y0_train[:,0])
y_train_min_U = np.min(y0_train[:,1])
y_train_max_U = np.max(y0_train[:,1])

x = (x0_train - x_train_max/2 - x_train_min/2) / (x_train_max/2 - x_train_min/2)
y = np.empty_like(y0_train)

y[:,0] = (y0_train[:,0] - y_train_max_I/2 - y_train_min_I/2) / (y_train_max_I/2 - y_train_min_I/2)
y[:,1] = (y0_train[:,1] - y_train_max_U/2 - y_train_min_U/2) / (y_train_max_U/2 - y_train_min_U/2)


del(x0_train,y0_train)


x_val = (x0_val - x_train_max/2 - x_train_min/2) / (x_train_max/2 - x_train_min/2)
y_val = np.empty_like(y0_val)
y_val[:,0] = (y0_val[:,0] - y_train_max_I/2 - y_train_min_I/2) / (y_train_max_I/2 - y_train_min_I/2)
y_val[:,1] = (y0_val[:,1] - y_train_max_U/2 - y_train_min_U/2) / (y_train_max_U/2 - y_train_min_U/2)

del(x0_val,y0_val)

x_test = (x0_test - x_train_max/2 - x_train_min/2) / (x_train_max/2 - x_train_min/2)
y_test = np.empty_like(y0_test)
y_test[:,0] = (y0_test[:,0] - y_train_max_I/2 - y_train_min_I/2) / (y_train_max_I/2 - y_train_min_I/2)
y_test[:,1] = (y0_test[:,1] - y_train_max_U/2 - y_train_min_U/2) / (y_train_max_U/2 - y_train_min_U/2)

del(x0_test,y0_test)

if use_saved_model == True:
    json_file = open(path_to_output+'saved\\'+ model_name +'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(path_to_output+'saved\\'+ model_name +'.h5')
    print("Loaded model from disk")
else:
    # define new keras model
    model = Sequential()
    model.add(Conv2D(32, (7, 7), input_shape=(dim[1],dim[0],1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(10, activation = 'sigmoid'))
    model.add(Dense(2))

# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Callbacks
#es = EarlyStopping(monitor='val_loss', patience=20)
csv_logger = CSVLogger(path_to_output+'checkpoint\\' + model_name + '_history_log.csv', append = True)

model_checkpoint_callback = ModelCheckpoint(
    filepath =  path_to_output+'checkpoint\\',
    save_weights_only = True,
    monitor = 'val_loss',
    mode = 'min',
    save_best_only = False)

if use_saved_model == False:
    epochs = 300
    bs = 32
    if shifts==True:
        train_DataGen = ImageDataGenerator(rotation_range = 5, width_shift_range = 0.2, height_shift_range = 0.2, fill_mode = 'nearest')
        val_DataGen = ImageDataGenerator()
        train_generator = train_DataGen.flow(
        x,y,
        batch_size=bs
        )
        validation_generator = val_DataGen.flow(
                x_val,y_val,
                batch_size=bs)
    
        # fit new keras model on the dataset
        with tf.device('/device:GPU:0'):
                history=model.fit_generator(
                train_generator,
                steps_per_epoch=max_train_num // bs,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=max_val_num // bs,
                callbacks = [csv_logger, model_checkpoint_callback])
    
    if shifts==False:
        with tf.device('/device:GPU:0'):
            history=model.fit(x,y, validation_data=(x_val, y_val), batch_size=bs, epochs=epochs, verbose=1, callbacks = [csv_logger, model_checkpoint_callback])
# evaluate the keras model


MSE = model.evaluate(x_val, y_val)
print('MSE: '+str(MSE))

if use_saved_model == False:
    plt.title('RMS error')
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('RMS error, rel.units')
    plt.legend()
    if save_model == True:
        plt.savefig( path_to_output+'saved\\'+ model_name +'.png')

# save model
if save_model == True:
    model_json = model.to_json()
    with open(path_to_output+'saved\\'+ model_name +'.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(path_to_output+'saved\\'+ model_name +'.h5')
    print('Saved model to disk')


# final tests

delta = np.empty((len(x_test),2))
delta0 = np.empty((len(x_test),2))

for i in range(0,len(x_test),1):
    y_predict = model.predict(x_test)
       
    delta[i,0] = y_test[i,0] - y_predict[i,0]
    delta[i,1] = y_test[i,1] - y_predict[i,1]

    delta0[i,0]=delta[i,0]*(y_train_max_I/2 - y_train_min_I/2)
    delta0[i,1]=delta[i,1]*(y_train_max_U/2 - y_train_min_U/2)
    
    
rms_I = np.sqrt(np.mean(delta0**2,0)[0])
rms_U = np.sqrt(np.mean(delta0**2,0)[1])

f = open(path_to_output+'saved\\'+ model_name+'_testing.txt','w')
f.write('RMS_I: '+str(rms_I/1e3)+'kA\n'+'RMS_U: '+str(rms_U)+' kV')
f.close()


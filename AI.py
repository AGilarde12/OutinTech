import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt 
import numpy as np
import random


dataDir = "Data/FER2013/train/"
classes = ["0", "1", "2", "3", "4", "5", "6"]

# #Resize image (we're doing transfer learning)
# img_size = 224 
# new_array = cv.resize(img_array, (img_size, img_size))
# plt.imshow(cv.cvtColor(new_array, cv.COLOR_BGR2RGB))
# plt.show()


#read all images and convert to array
training_data = []
img_size = 224 #mobileNetV2 takes 224x224

def createTrainingData():
    for category in classes:
        path = os.path.join(dataDir, category)
        classNum = classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path,img))
                new_array = cv.resize(img_array, (img_size, img_size))
                training_data.append([new_array, classNum])
            except Exception as e:
                pass

createTrainingData()

#print(len(training_data))

random.shuffle(training_data) #So training is dynamic

x = [] #features
y = [] #labels

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, img_size, img_size, 3) #convert to 4D
y = np.array(y)

#print(x.shape)

#normalize data
x = x/255.0

#training model - transfer learning
model = tf.keras.applications.MobileNetV2() #pretrained model

#model.summary()

#transfer learning
base_input = model.layers[0].input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output) #add new layer after output of global pooling layer
final_output = layers.Activation('relu')(final_output) #activation function
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output) #classification layer

new_model = keras.Model(inputs = base_input, outputs = final_output)

#new_model.summary()

new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#train and save Model
new_model.fit(x,y, epochs=30)
new_model.save('Emo_recV0.h5')

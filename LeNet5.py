#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# import cv2

# model as sequential model
model = Sequential()
# add convolution layer (32x32x1)
model.add(Convolution2D(filters=6,kernel_size=5,strides=1,activation='relu',input_shape=(32,32,1)))

# add max pooling layer (28x28x6)
model.add(MaxPooling2D(pool_size=2,strides=2))

# (14x14x6)
model.add(Convolution2D(filters=16,kernel_size=5,strides=1,activation='relu',input_shape=(14, 14, 6)))

# (10x10x16)
model.add(MaxPooling2D(pool_size=2,strides=2))

# add flatten layer (5x5x16)
model.add(Flatten())
# add FC layer (400)
model.add(Dense(120,activation='relu', input_dim=400))
model.add(Dense(84,activation='relu', input_dim=120))
model.add(Dense(20,activation='softmax', input_dim=84))
# compile
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

from keras.optimizers import Adam
optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# training
data = []
labels = []

import os
from scipy import signal

labelnames = ["Tettigonioidea1", "Tettigonioidea2", "drums_Snare", "Grylloidea1",\
 "drums_MidTom", "drums_HiHat", "drums_Kick", "drums_SmallTom",\
  "guitar_chord2", "Frog1", "Frog2", "drums_FloorTom", "guitar_7th_fret", \
  "drums_Rim", "Grylloidea2", "guitar_3rd_fret", "drums_Ride", \
  "guitar_chord1", "guitar_9th_fret", "Frog3"]

for (lbl,name) in enumerate(labelnames):
	tdir = "train/{}/".format(name)
	files = os.listdir(tdir)
	for filename in files:
		sig = np.load(tdir + filename)
		f, t, gram = signal.spectrogram(sig, window=signal.get_window("hamming", 63))#63->32
		# print(f)
		# gram = cv2.normalize(gram, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		print(gram.shape)
		# to (32x32)
		data.append(gram)
		label.append([float(i==lbl) for i in range(20)])
		

model.fit(x=data,y=label,epochs=20,batch_size=32)
# data is a tensor with shape (# of data, height, width, depth)
# label is a tensor with shape (# of labels, # of classes)
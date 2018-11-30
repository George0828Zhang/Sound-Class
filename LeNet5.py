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
i_sz = 64
sz = i_sz
ks = 5
# add convolution layer (32x32x1)
model.add(Convolution2D(filters=6,kernel_size=ks,strides=1,activation='relu',input_shape=(sz,sz, 1)))

# add max pooling layer (28x28x6)
model.add(MaxPooling2D(pool_size=2,strides=2))

# (14x14x6)
sz = (sz-ks+1)/2
model.add(Convolution2D(filters=16,kernel_size=ks,strides=1,activation='relu',input_shape=(sz, sz, 6)))

# (10x10x16)
model.add(MaxPooling2D(pool_size=2,strides=2))

# add flatten layer (5x5x16)
sz = (sz-ks+1)/2
model.add(Flatten())
# add FC layer (400)
sz = sz**2*16
model.add(Dense(120,activation='relu', input_dim=sz))

model.add(Dense(84,activation='relu', input_dim=120))

model.add(Dense(20,activation='softmax', input_dim=84))



from keras.optimizers import Adam
optimizer = Adam(lr=0.0001)
# from keras.optimizers import SGD
# optimizer = SGD(lr=0.0001, momentum=0.2, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# training
data = []
labels = []
v_data = []
v_labels = []

import os
from scipy import signal, misc

labelnames = ["Tettigonioidea1", "Tettigonioidea2", "drums_Snare", "Grylloidea1",\
"drums_MidTom", "drums_HiHat", "drums_Kick", "drums_SmallTom",\
"guitar_chord2", "Frog1", "Frog2", "drums_FloorTom", "guitar_7th_fret", \
"drums_Rim", "Grylloidea2", "guitar_3rd_fret", "drums_Ride", \
"guitar_chord1", "guitar_9th_fret", "Frog3"]

import math
def opt_width(sig):
	w = math.floor(((len(sig)*112+1)**0.5-1)/7.)
	return w if w % 2 == 0 else (w + 1)
def refine(sig, w):
	h = w//2
	l = 2*h**2-(h-1)*(h//4)
	print(l, len(sig))
	if l > len(sig):
		return np.pad(sig, (0, l-len(sig)), 'constant')
	else:
		return sig[0:l]

def sig2data(sig):
	width = opt_width(sig)
	hwindow=signal.get_window("hamming", width)
	# what about phase?
	f, t, gram = signal.spectrogram(sig, window=hwindow, nperseg=width, mode="complex")
	# downscale to (nxn), 'lanczos' or 'bicubic'
	# this requires "pillow" pkg
	gram = misc.imresize(gram, (i_sz, i_sz), interp='lanczos', mode=None)
	return np.expand_dims(gram, axis=2)

for (lbl,name) in enumerate(labelnames):
	tdir = "train/{}/".format(name)
	tfiles = os.listdir(tdir)
	for filename in tfiles:
		sig = np.load(tdir + filename)		
		data.append(sig2data(sig))
		labels.append([float(i==lbl) for i in range(20)])

	vdir = "val/{}/".format(name)
	vfiles = os.listdir(vdir)
	for filename in vfiles:
		sig = np.load(vdir + filename)
		v_data.append(sig2data(sig))
		v_labels.append([float(i==lbl) for i in range(20)])
		

model.fit(x=np.asarray(data),y=np.asarray(labels),epochs=30,batch_size=64)
# data is a tensor with shape (# of data, height, width, depth)
# label is a tensor with shape (# of labels, # of classes)

loss, metrics = model.evaluate(np.asarray(v_data), np.asarray(v_labels), batch_size=128)
print ("Loss={:.5f} Accu={:.2f}%".format(loss,metrics*100))

if 'y' in input("Save model? [y/n]"):
	filename = input("filename:")
	if filename[-3:] != ".h5":
		filename += ".h5"
	model.save(filename)

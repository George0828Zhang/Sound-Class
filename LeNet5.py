#!/usr/bin/env python3

import numpy as np
import sys
import os
import math
# import cv2

i_sz = 64

data = []
labels = []
v_data = []
v_labels = []
if len(sys.argv) != 3:
	print("expected 2 arguments: [trainfolder] [valfolder]")
	exit(1)
elif os.path.isdir(sys.argv[1]) and os.path.isdir(sys.argv[2]):
	pass
else:
	print("cannot parse arguments, please retry.")
	exit(1)

# training

from scipy import signal
labelnames = ["Tettigonioidea1", "Tettigonioidea2", "drums_Snare", "Grylloidea1",\
"drums_MidTom", "drums_HiHat", "drums_Kick", "drums_SmallTom",\
"guitar_chord2", "Frog1", "Frog2", "drums_FloorTom", "guitar_7th_fret", \
"drums_Rim", "Grylloidea2", "guitar_3rd_fret", "drums_Ride", \
"guitar_chord1", "guitar_9th_fret", "Frog3"]

data_cnt = 0
processed_cnt = 0
for (lbl,name) in enumerate(labelnames):
	tdir = "spec1/{}/".format(name)
	tfiles = os.listdir(tdir)	
	vdir = "spec2/{}/".format(name)
	vfiles = os.listdir(vdir)
	data_cnt += len(tfiles) + len(vfiles)

def showprogress(showbar=True):
	bar = ""
	if showbar:
		cnt = int(processed_cnt/data_cnt*30)
		bar = "["+"="*30+"]" if cnt==30 else ("["+"="*cnt+">"+"."*(29-cnt)+"]")
	print("Processing data... ({}/{}) {}".format(processed_cnt,data_cnt, bar), end='\r')

if len(data) == 0:
	showprogress()
	for (lbl,name) in enumerate(labelnames):
		tdir = "spec1/{}/".format(name)
		tfiles = os.listdir(tdir)
		for filename in tfiles:
			sig = np.load(tdir + filename)		
			data.append(sig2data2(sig))
			labels.append([float(i==lbl) for i in range(20)])
			processed_cnt += 1
			 np.expand_dims(gram, axis=2)
			showprogress()

		vdir = "spec2/{}/".format(name)
		vfiles = os.listdir(vdir)
		for filename in vfiles:
			sig = np.load(vdir + filename)
			v_data.append(sig2data2(sig))
			v_labels.append([float(i==lbl) for i in range(20)])
			processed_cnt += 1
			showprogress()
	print("")	
	print("Saving spectrogram data to .npy files...")
	np.save("train_data.npy", data)
	np.save("train_label.npy", labels)
	np.save("val_data.npy", v_data)
	np.save("val_label.npy", v_labels)
	print("Done.")







from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# model as sequential model
model = Sequential()

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

model.fit(x=np.asarray(data),y=np.asarray(labels),epochs=22,batch_size=32)
# data is a tensor with shape (# of data, height, width, depth)
# label is a tensor with shape (# of labels, # of classes)

loss, metrics = model.evaluate(np.asarray(v_data), np.asarray(v_labels), batch_size=128)
print ("Loss={:.5f} Accu={:.2f}%".format(loss,metrics*100))

if 'y' in input("Save model? [y/n]"):
	filename = input("filename:")
	if filename[-3:] != ".h5":
		filename += ".h5"
	model.save(filename)

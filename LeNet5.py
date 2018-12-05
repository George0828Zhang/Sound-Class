#!/usr/bin/env python3

import numpy as np
import sys
import os
import math
import cv2

sz_x = 64
sz_y = 64

data = []
labels = []
v_data = []
v_labels = []
if len(sys.argv) != 4:
	print("expected 3 arguments: [trainfolder] [valfolder] [save.h5/none]")
	exit(1)
elif os.path.isdir(sys.argv[1]) and os.path.isdir(sys.argv[2]):
	trainF = sys.argv[1] 
	valF = sys.argv[2]
	if trainF[-1] == '/':
		trainF = trainF[:-1]
	if valF[-1] == '/':
		valF = valF[:-1]
	savefile = sys.argv[3]
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
	tdir = "{}/{}/".format(trainF,name)
	tfiles = os.listdir(tdir)	
	vdir = "{}/{}/".format(valF,name)
	vfiles = os.listdir(vdir)
	data_cnt += len(tfiles) + len(vfiles)

def showprogress(showbar=True):
	bar = ""
	if showbar:
		cnt = int(processed_cnt/data_cnt*30)
		bar = "["+"="*30+"]" if cnt==30 else ("["+"="*cnt+">"+"."*(29-cnt)+"]")
	print("Processing data... ({}/{}) {}".format(processed_cnt,data_cnt, bar), end='\r', flush=True)

showprogress()
for (lbl,name) in enumerate(labelnames):
	tdir = "{}/{}/".format(trainF,name)
	tfiles = os.listdir(tdir)
	for filename in tfiles:
		gram = cv2.imread(tdir+filename, cv2.IMREAD_GRAYSCALE)
		# data.append(gram)
		data.append(np.expand_dims(gram, axis=2))
		labels.append([float(i==lbl) for i in range(20)])
		processed_cnt += 1		
		showprogress()

	vdir = "{}/{}/".format(valF,name)
	vfiles = os.listdir(vdir)
	for filename in vfiles:
		gram = cv2.imread(vdir+filename, cv2.IMREAD_GRAYSCALE)
		# v_data.append(gram)
		v_data.append(np.expand_dims(gram, axis=2))
		v_labels.append([float(i==lbl) for i in range(20)])
		processed_cnt += 1
		showprogress()
print("\r\nDone")



from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
# model as sequential model
model = Sequential()

ks = 5
# add convolution layer (32x32x1)
model.add(Convolution2D(filters=6,kernel_size=ks,strides=1,activation='relu',input_shape=(sz_x,sz_y, 1)))

# add max pooling layer (28x28x6)
model.add(MaxPooling2D(pool_size=2,strides=2))
# model.add(BatchNormalization())

# (14x14x6)
model.add(Convolution2D(filters=16,kernel_size=ks,strides=1,activation='relu'))

# (10x10x16)
model.add(MaxPooling2D(pool_size=2,strides=2))
# model.add(BatchNormalization())

# add flatten layer (5x5x16)
model.add(Flatten())
# add FC layer (400)
model.add(Dense(120,activation='relu'))
# model.add(Dropout(0.35))
# model.add(BatchNormalization())

model.add(Dense(84,activation='relu'))
# model.add(Dropout(0.35))
# model.add(BatchNormalization())

model.add(Dense(20,activation='softmax'))



batch_size = 256
lr = 0.001*batch_size/64
# about 90 is optimal
epoch = 90

# get params
# fp = open("config.cfg", "rt")
# params = [int(x) for x in fp.read().split()]
# batch_size, epoch = params[0:2]


from keras.optimizers import Adam
optimizer = Adam(lr=lr)
# from keras.optimizers import SGD
# optimizer = SGD(lr=0.0001, momentum=0.2, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

model.fit(x=np.asarray(data),y=np.asarray(labels),epochs=epoch,batch_size=batch_size)
# data is a tensor with shape (# of data, height, width, depth)
# label is a tensor with shape (# of labels, # of classes)


loss, metrics = model.evaluate(np.asarray(v_data), np.asarray(v_labels), batch_size=128)

print ("Loss={:.5f} Accu={:.2f}%".format(loss,metrics*100))

if savefile not in ["None", "none", "null"]:
	if savefile[-3:] != ".h5":
		savefile += ".h5"
	print("Saving model to "+savefile+"...")
	model.save(savefile)

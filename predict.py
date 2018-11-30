#!/usr/bin/env python3

import numpy as np
import sys
import os
import math
import cv2
from GenSpectrogram import sig2data
from keras.models import load_model

if len(sys.argv) != 4:
	print("expected 3 arguments: [model.h5] [test.npy] [save.npy/none]")
	exit(1)
elif os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]):
	model = load_model(sys.argv[1])
	sigdata = np.load(sys.argv[2])
	savefile = sys.argv[3]
else:
	print("cannot parse arguments, please retry.")
	exit(1)

# testing

data_cnt = len(sigdata)
processed_cnt = 0

def showprogress(showbar=True):
	bar = ""
	if showbar:
		cnt = int(processed_cnt/data_cnt*30)
		bar = "["+"="*30+"]" if cnt==30 else ("["+"="*cnt+">"+"."*(29-cnt)+"]")
	print("Processing data... ({}/{}) {}".format(processed_cnt,data_cnt, bar), end='\r', flush=True)

data = []

showprogress()
for x in sigdata:
	gram = np.expand_dims(sig2data(x), axis=2)
	data.append(gram)
	processed_cnt += 1
	showprogress()
print("\r\nDone")

Px = model.predict(np.asarray(data), batch_size=32, verbose=1, steps=None)

predictions = []
for probs in Px:
	lbl = np.argmax(probs)
	predictions.append(lbl)

# print(predictions)	
if savefile not in ["None", "none", "null"]:
	if savefile[-4:] != ".npy":
		savefile += ".npy"
	print("Saving results to "+savefile+"...")
	np.save(savefile, predictions)

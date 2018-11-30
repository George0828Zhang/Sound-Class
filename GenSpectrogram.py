#!/usr/bin/env python3
import cv2
import math	
from scipy import signal

i_sz = 64

def sig2data(sig):
	width = 128#math.floor(((len(sig)*112+1)**0.5-1)/7.)
	hwindow=signal.get_window("hamming", width)
	f, t, gram = signal.spectrogram(sig, window=hwindow, nperseg=width, mode="magnitude")
	gram = cv2.normalize(gram, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	gram = cv2.resize(gram, (i_sz, i_sz), interpolation=cv2.INTER_LANCZOS4)#CUBIC
	return gram

if __name__ == "__main__":

	import numpy as np
	import sys
	import os

	if len(sys.argv) != 3:
		print("expected 2 arguments: [trainfolder] [valfolder]")
		exit(1)
	elif os.path.isdir(sys.argv[1]) and os.path.isdir(sys.argv[2]):	
		trainF = sys.argv[1] 
		valF = sys.argv[2]
		if trainF[-1] == '/':
			trainF = trainF[:-1]
		if valF[-1] == '/':
			valF = valF[:-1]
	else:
		print("cannot parse arguments, please retry.")
		exit(1)

	# training	
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
		print("Processing data... ({}/{}) {}".format(processed_cnt,data_cnt, bar), end='\r')

	def createdir(sdir):
		if not os.path.exists(sdir):
			os.makedirs(sdir)

	showprogress()
	for (lbl,name) in enumerate(labelnames):
		tdir = "{}/{}/".format(trainF,name)
		odir = "{}/{}/".format(trainF+"2",name)
		createdir(odir)
		tfiles = os.listdir(tdir)
		for filename in tfiles:
			sig = np.load(tdir + filename)		
			img = sig2data(sig)
			cv2.imwrite(odir+filename+".png", img)
			processed_cnt += 1
			showprogress()

		vdir = "{}/{}/".format(valF,name)
		odir = "{}/{}/".format(valF+"2",name)
		createdir(odir)
		vfiles = os.listdir(vdir)
		for filename in vfiles:
			sig = np.load(vdir + filename)
			img = sig2data(sig)
			cv2.imwrite(odir+filename+".png", img)
			processed_cnt += 1
			showprogress()
	print("")
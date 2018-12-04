BASE=result97.npy
RES=results.npy
MOD=model.h5
NET=LeNet5.py

all: spectro train predict
comp: comp.py $(RES) $(BASE)
	python3 comp.py $(RES) $(BASE) 
predict: $(MOD) test.npy predict.py
	python3 predict.py $(MOD) test.npy $(RES)
train: train2/ val2/ $(NET)
	python3 $(NET) train2/ val2/ $(MOD)
spectro: train/ val/ GenSpectrogram.py
	python3 GenSpectrogram.py train/ val/
clean:
	rm -rf train2/
	rm -rf val2/
	rm $(MOD)
	rm $(RES)
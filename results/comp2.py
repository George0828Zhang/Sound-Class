#! /usr/bin/env python3

import numpy as np
import sys

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("expected at least 2 arguments [src1] [src2] ...")
		exit(1)
	results = []
	signi = []
	for file in sys.argv[1:]:
		ans = np.load(file)
		results.append(ans)
		signi.append(1)
	samples = len(results[0])
	finalans = [-1]*samples
	for i in range(samples):
		# vote
		# vote = np.asarray(results)[:,i]
		vote = [0]*20
		for anss in results:
			lbl = anss[i]
			vote[lbl] += 1
		ans = max(vote)
		# check if global max
		if vote.count(ans)==1:
			finalans[i] = vote.index(ans)
			for (j,anss) in enumerate(results):
				if anss[i] == finalans[i]:
					signi[j] += 1
	boss = signi.index(max(signi))
	for i in range(samples):
		if finalans[i] == -1:
			finalans[i] = results[boss][i]

	print("Total diffs:")
	for i in range(len(sys.argv)-1):
		diffs = sum([p!=q for (p,q) in zip(results[i], finalans)])
		print("{} {}/{} ({:.2f}%)".format(sys.argv[i+1], diffs, samples, diffs*100/samples))

	np.save("results.npy", finalans)
	print("Final result saved to results.npy")


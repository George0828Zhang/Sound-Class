#! /usr/bin/env python3

import numpy as np
import sys

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("expected at least 2 arguments [src1] [src2] ...")
		exit(1)
	results = []
	for file in sys.argv[1:]:
		ans = np.load(file)
		results.append(ans)
	diffs = 0
	for (i, lb) in enumerate(results[0]):
		notice = False
		anss = [lb]
		for dif in results[1:]:
			myans = dif[i]
			if myans != lb:
				notice = True
			anss.append(myans)
		if notice:
			diffs += 1
			print("No: {} predicts: {}".format(i, anss))
	print("Total diffs: {} out of {} ({:.2f}%)".format(diffs, len(results[0]), diffs*100/len(results[0])))


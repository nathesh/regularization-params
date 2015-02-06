import csv 
import numpy as np 
import matplotlib.pyplot as P
import os 
from os import path 

dirc = '/home/thejas/Documents/Python/regularization-params/output'
files = os.listdir(dirc)
for fi in files:
#for c in range(3,-4,-1):
	#alpha = 10**c
	alpha = fi.split('_')[1].split('.csv')[0]
	#n = "output/alpha_" + str(alpha) + ".csv"
	with open(fi,'r') as f:
		spamreader = csv.reader(f, delimiter=' ', quotechar='|')
		f1 = []
		accuracy = []
		precision = []
		recall = []
		d = 0
		for row in spamreader:
			if d != 0:
				f,a,p,r = row[0].split(',')
				f1.append(float(f))
				accuracy.append(float(a))
				precision.append(float(p))
				recall.append(float(r))
			else:
				d +=1

		f1 = np.array(f1)
		accuracy = np.array(accuracy)
		precision = np.array(precision)
		recall = np.array(recall)
		f, ax = P.subplots(4,sharex=True)
		title = 'Alpha ' + str(alpha)
		P.suptitle(title)
		ax[0].hist(f1,range=(0,1))
		ax[0].set_title('F1 Measure')

		ax[1].hist(accuracy,range=(0,1))
		ax[1].set_title('accuracy')

		ax[2].hist(precision,range=(0,1))
		ax[2].set_title('precision')

		ax[3].hist(recall,range=(0,1))
		ax[3].set_title('recall')
		out = "results/alpha_" + str(alpha) + ".png"
		P.savefig(out)

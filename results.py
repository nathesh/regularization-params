import csv 
import numpy as np 
import matplotlib.pyplot as P
import os 
from os import path 
import seaborn as sns 	
dirc = '/home/thejas/Documents/Python/regularization-params/op'
files = os.listdir(dirc)
for fi in files:
#for c in range(3,-4,-1):
	#alpha = 10**c
	alpha = fi.split('_')[1].split('.csv')[0]
	#n = "output/alpha_" + str(alpha) + ".csv"
	fip = dirc  + "/" + fi
	with open(fip,'r') as f:
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
		f, ax1 = P.subplots(4,figsize=(8, 8),sharex=True)
		title = 'Alpha ' + str(alpha)
		P.suptitle(title)
		sns.kdeplot(f1,shade=True,ax=ax1[0])
		#ax[0].hist(f1,range=(.75,.98))
		ax1[0].set_title('F1 Measure')

		sns.kdeplot(accuracy,ax=ax1[1])
		#ax[1].hist(accuracy,range=(.75,.98))
		ax1[1].set_title('accuracy')

		sns.kdeplot(precision,ax=ax1[2])
		#ax[2].hist(precision,range=(.75,.98))
		ax1[2].set_title('precision')

		sns.kdeplot(recall,ax=ax1[3])
		#ax[3].hist(recall,range=(.75,.98))
		ax1[3].set_title('recall')
		out = "results_focus_2/alpha_" + str(alpha) + ".png"
		P.savefig(out)
		
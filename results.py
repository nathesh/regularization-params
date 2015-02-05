import csv 
import numpy as np 
import matplotlib.pyplot as P


for c in range(3,-4,-1):
	alpha = 10**c
	n = "output/alpha_" + str(alpha) + ".csv"
	with open(n,'r') as f:
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
		f, ax = P.subplots(2,2)
		title = 'Alpha ' + str(alpha)
		P.suptitle(title)
		ax[0,0].hist(f1)
		ax[0,0].set_title('F1 Measure')

		ax[0,1].hist(accuracy)
		ax[0,1].set_title('accuracy')

		ax[1,0].hist(precision)
		ax[1,0].set_title('precision')

		ax[1,1].hist(recall)
		ax[1,1].set_title('recall')
		out = "results/alpha_" + str(alpha) + ".png"
		P.savefig(out)

import csv 
import numpy as np 
import matplotlib.pyplot as P

with open("output/alpha_1.csv",'r') as f:
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
	print precision, recall 
	P.hist(f1)
	P.show()
	P.hist(accuracy)
	P.show()
	P.hist(precision)
	P.show
	P.hist(recall)
	P.show

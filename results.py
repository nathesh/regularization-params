import csv 
import numpy as np 
import matplotlib.pyplot as P
import os 
from os import path 
import seaborn as sns 	
def check(x):
	if x == 0:
		return "F1"
	elif x == 1:
		return "accuracy"
	elif x == 2: 
		return "precision"
	elif x == 3:
		return "recall"
	else:
		return str(0)
dirc = '/home/thejas/Documents/Python/regularization-params/op'
files = os.listdir(dirc)
fil = list(enumerate(files))
f, axf = P.subplots(5,4,sharex=True)
P.suptitle("F1")	
f1_t = []
acc_t = []
prec_t = []
rec_t = []
for nu,fi in fil:
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

		f1 		  = np.array(f1)
		accuracy  = np.array(accuracy)
		precision = np.array(precision)
		recall 	  = np.array(recall)
		f1_t.append(f1)
		acc_t.append(accuracy)
		prec_t.append(precision)
		rec_t.append(recall)
		f1_mean = np.mean(f1)
		accuracy_mean = np.mean(accuracy)
		precision_mean = np.mean(precision)
		recall_mean = np.mean(recall)
		print f1_mean
		# Focus_3 
		'''
		axf[nu%5][nu/5].set_title('Alpha ' + str(alpha))
		axa[nu%5][nu/5].set_title('Alpha ' + str(alpha))
		axp[nu%5][nu/5].set_title('Alpha ' + str(alpha))
		axr[nu%5][nu/5].set_title('Alpha ' + str(alpha))
		'''
		
		sns.kdeplot(f1,ax=axf[nu%5][nu/5])
		'''
		sns.kdeplot(accuracy,ax=axa[nu%5][nu/5])
		sns.kdeplot(precision,ax=axf[nu%5][nu/5])
		sns.kdeplot(recall,ax=axf[nu%5][nu/5])
		'''
		# This for focus_2
		'''
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
		'''
'''
out = "results_focus_3/recall.jpg"
P.savefig(out)
out = "results_focus_3/precision.jpg"
P.savefig(out)
out = "results_focus_3/accuracy.jpg"
P.savefig(out)'''
out = "results_focus_3/F1.jpg"
P.savefig(out)
All = (f1_t,acc_t,prec_t,rec_t)
for x in range(0,4):
	now = All[x]
	f, axf = P.subplots(5,4,sharex=True)
	P.suptitle(check(x))
	for y in range(0,20):
		cu = now[y]
		sns.kdeplot(cu,ax=axf[y%5][y/5])
	out = "results_focus_3/" + check(x) + ".jpg"
	P.savefig(out)

def check(x):
	if x == 0:
		return "F1"
	elif x == 1:
		return "accuracy"
	elif x == 2: 
		return "precision"
	elif x == 3:
		return "recall"
	else:
		return str(0)
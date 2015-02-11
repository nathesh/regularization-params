import csv 
import numpy as np 
import matplotlib.pyplot as P
import os 
from os import path 
import seaborn as sns 	
import matplotlib.text as txt
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
files.sort()
fil = list(enumerate(files))
f, axf = P.subplots(5,4,sharex=True)
P.suptitle("F1")	
f1_t        	=   []
acc_t      		=   []
prec_t     		=   []
rec_t      		=   []
Alpha_vals		=	[]
f1_mean_t   	=   []
acc_mean_t  	=   []
prec_mean_t 	=   []
rec_mean_t  	=   []
f1_precentile	=	[]
acc_precentile	=	[]
prec_precentile	=	[]
rec_precentile	=	[]
for nu,fi in fil:
#for c in range(3,-4,-1):
	#alpha = 10**c
	alpha = fi.split('_')[1].split('.csv')[0]
	#n = "output/alpha_" + str(alpha) + ".csv"
	fip = dirc  + "/" + fi
	Alpha_vals.append(float(alpha))
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
		f1_mean_t.append(f1_mean)
		acc_mean_t.append(accuracy_mean)
		prec_mean_t.append(precision_mean)
		rec_mean_t.append(recall_mean)
		f1_precentile.append((np.percentile(f1,25),np.percentile(f1,75)))
		acc_precentile.append((np.percentile(accuracy,25),np.percentile(accuracy,75)))
		prec_precentile.append((np.percentile(precision,25),np.percentile(precision,75)))
		rec_precentile.append((np.percentile(recall,25),np.percentile(recall,75)))
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
All 			= (f1_t,acc_t,prec_t,rec_t)
All_mean 		= (f1_mean_t,acc_mean_t,prec_mean_t,rec_mean_t)
All_precentiles = (f1_precentile,acc_precentile,prec_precentile,rec_precentile)
for x in range(0,4):
	now 		= All[x]
	f, axf 		= P.subplots(5,4,figsize=(16,16),sharex='all',sharey='all',squeeze=False)
	P.suptitle(check(x))
	P.tight_layout()
	now_mean 	= All_mean[x]
	now_precent = All_precentiles[x]
	for y in range(0,20):
		cu = now[y]
		sns.kdeplot(cu,ax=axf[y%5][y/5],clip=(.75,.9))
		axf[y%5][y/5].axvline(now_mean[y], ls="--", linewidth=1.5)
		axf[y%5][y/5].axvline(now_precent[y][0], ls="-", linewidth=1.5,color="black")
		axf[y%5][y/5].axvline(now_precent[y][1], ls="-", linewidth=1.5,color="black")
		axf[y%5][y/5].set_title("C="+"%.3f" % (float(Alpha_vals[y])**-1))
		text = '$\mu=%.2f$\n$\mathrm{25}=%.2f$\n$\mathrm{75}=%.2f$'%(float(now_mean[y]), float(now_precent[y][0]),float(now_precent[y][1]))
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		axf[y%5][y/5].text(0.05, 0.95, text, transform=axf[y%5][y/5].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
        print 'Done!'
		#axf[y%5][y/5].set_title("C="+str(Alpha_vals[y]))
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
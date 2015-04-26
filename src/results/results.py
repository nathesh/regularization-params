import csv 
import numpy as np 
import matplotlib.pyplot as P
import os 
from os import path 
import seaborn as sns 	
import matplotlib.text as txt
import pdb 
from cost_function import cost_function
import crossv 
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

loss_types = ["log","hinge"]
vote_types = ["MAX"]
for loss_type in loss_types:
	for vote_type in vote_types:
		print 'RUNNING', loss_type,vote_type
		dirc = '../../output/irony/CL/' + loss_type + '/' + vote_type + '/'
		files = os.listdir(dirc)
		files.remove('results')
		files_1 = sorted(files,key = lambda x: int(x.split('_')[1]))
		files_1 = files_1[:6]
		#print files_1
		files_1 = list(enumerate(files_1,start=1))
		for results_1,files in files_1: # set of 20 trails i.e. trails_1
			
			#read_num = reader.next()
			
			files = dirc + files + "/"
			files_2 = files
			files = os.listdir(files)
			files = sorted(files,key=lambda x:float(x.split('.csv')[0])**-1)
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
			for nu,fi in fil: # alpha value 
			#for c in range(3,-4,-1):
				#alpha = 10**c
				#	print fi
				alpha = fi.split('.csv')[0]
				#n = "output/alpha_" + str(alpha) + ".csv"
				fip = files_2  + "/" + fi
				Alpha_vals.append(float(alpha))
				with open(fip,'r') as f: # Open up for the alpha value 
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

			All 			= (f1_t,acc_t,prec_t,rec_t)
			All_mean 		= (f1_mean_t,acc_mean_t,prec_mean_t,rec_mean_t)
			All_precentiles = (f1_precentile,acc_precentile,prec_precentile,rec_precentile)
			
			for x in range(0,4):
				print check(x),results_1
				#pdb.set_trace()
				#f, axf 		= P.subplots(5,4,figsize=(16,16),sharex='all',sharey='all',squeeze=True)	
				num_vals = range(0,20)
				now 		= All[x]
				num = 5
				f, axf 		= P.subplots(5,4,figsize=(16,16),sharex='all',sharey='all',squeeze=False)
				P.suptitle(check(x))
				P.tight_layout()
				sns.set_context("paper")
				#P.xlim([0,1])
				now_mean 	= All_mean[x]
				now_precent = All_precentiles[x]
				for y in num_vals:
					cv = crossv.run(Alpha_vals[y],vote_type,loss_type)
					cu = now[y]
					#print 'y =',y,'min value = ', np.min(cu),'max value = ', np.max(cu),'mean =', now_mean[y]
					if np.min(cu) == 0:
						continue  
					if np.min(cu) <= 0 or np.max(cu) > 1:
						print 'There are negs!'
					#cost_trail = cost_function_i.get_cost(y)
					sns.kdeplot(cu,ax=axf[y%num][y/num]) # throwing an error at 12?
					axf[y%num][y/num].axvline(now_mean[y], ls="--", linewidth=1.5)
					axf[y%num][y/num].axvline(cv[x], ls="--", linewidth=25,color="red")
					axf[y%num][y/num].axvline(now_precent[y][0], ls="-", linewidth=1.25,color="black")
					axf[y%num][y/num].axvline(now_precent[y][1], ls="-", linewidth=1.25,color="black")
					axf[y%num][y/num].set_title("C="+"%3.4g" % (float(Alpha_vals[y])**-1))
					text = '$\mu=%.2f$\n(%.2f,%.2f)'%(float(now_mean[y]), float(now_precent[y][0]),float(now_precent[y][1]))
					props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
					axf[y%num][y/num].text(0.05, 0.95, text, transform=axf[y%num][y/num].transAxes, fontsize=12,
					verticalalignment='top', bbox=props)
					#print 'Done!'
					#axf[y%5][y/5].set_title("C="+str(Alpha_vals[y]))
				out = '../../output/irony/CL/' + loss_type + '/' + vote_type + '/results/trails_' + str(results_1) + '_results/' + check(x) + ".jpg"
				P.savefig(out)
				P.close(out)

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

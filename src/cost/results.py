import numpy
import os
import csv
import numpy as np
from numpy.random import randn
from cost_function import cost_function
import seaborn as sns 
import matplotlib.pyplot as P
from pylab import *
import matplotlib.patches as mpatches
from matplotlib.ticker import OldScalarFormatter
from matplotlib.ticker import ScalarFormatter 
from bootstraps import bootstrap 
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
def LT(loss_type):
	if loss_type == 0:
		return 'log'
	else:
		return 'hinge'

def write():
	loss_types = ['log', 'hinge']
	with open("cost.csv", 'w') as out:
		csv_out = csv.writer(out)
		##csv_out.writerow(('loss type','trail number','F1 max', 'F1 min', 'Accuracy max', 'Accuracy min','Precision max','Precision min','Recall max','Recall min'))
		for loss_types in loss_types:
			dirc = '../../output/irony/CL/' + loss_type + '/MAX/'
			files = os.listdir(dirc)
			files.remove('results')
			files_1 = sorted(files, key=lambda x: int(x.split('_')[1]))
			files_1 = files_1[:6]
			counter_files = 0
			for file in files_1:  # these are the 6 trails (set of 6 trails)
				current_trail = dirc + file + '/'
				current_files = os.listdir(current_trail)
				current_files = sorted(current_files, key=lambda x: float(x.split('.csv')[0]) ** -1)
				f1_t = []
				acc_t = []
				prec_t = []
				rec_t = []
				for current_file in current_files:  # 20 trails in that 6 trails
					alpha = current_file.split('.csv')[0]
					open_file = current_trail + current_file
					with open(open_file, 'r') as f:  # this for each trail 20 of these 
						spamreader = csv.reader(f, delimiter=' ', quotechar='|')
						counter = 0
						f1 = []
						accuracy = []
						precision = []
						recall = []
						for row in spamreader:
							if counter != 0:
								f, a, p, r = row[0].split(',')
								f1.append(float(f))
								accuracy.append(float(a))
								precision.append(float(p))
								recall.append(float(r))
							else:
								counter += 1

					f1_t.append(np.array(f1))
					acc_t.append(np.array(accuracy))
					prec_t.append(np.array(precision))
					rec_t.append(np.array(recall))
				counter_files = counter_files + 1
				write = (loss_type,counter_files,np.max(f1_t),np.percentile(f1_t,33),\
						np.max(acc_t),np.percentile(acc_t,33),np.max(prec_t),\
						np.percentile(prec_t,33),np.max(rec_t),np.percentile(rec_t,33))
				csv_out.writerow(write)

				print "loss type: ", loss_type, counter_files+1
				print "\nF1: ", np.max(f1_t),   np.percentile(f1_t, 33)
				print "Accuracy: ", np.max(acc_t),  np.percentile(acc_t, 33)
				print "Precision: ", np.max(prec_t), np.percentile(prec_t, 33)
				print "Recall: ", np.max(rec_t),  np.percentile(rec_t, 33)
				print "\n"

def cost_function_graphs():
	cost_function_i = cost_function()
	with open("test.csv", 'r') as cost_file:
		cost_reader = csv.reader(cost_file, delimiter=',', quotechar='|')
		loss_types = ['log','hinge']
		for loss_num, loss_type in enumerate(loss_types):
			bootstraps_t = bootstrap()
			alpha_values = []
			path = '../../output/irony/CL/' + loss_type + '/MAX/'
			trails_files = os.listdir(path)
			trails_files.remove('results')	
			trails_files = sorted(trails_files, key=lambda x: int(x.split('_')[1]))
			trails_files = [path + file+"/" for file in trails_files]
			trails_files = trails_files[:6]
			for trails_file in trails_files: 
				cost_read = cost_reader.next() # this is the cost for 20 bootstrap trails (100)
				cost_read = cost_read[2:]
				cost_read = [float(cr) for cr in cost_read] 				
				cost_read =  [(cost_read[x],cost_read[x+1]) for x in range(0,len(cost_read),2)]
				#print cost_read
				cost_function_i.add_min_max(cost_read)
				alpha_trails = os.listdir(trails_file)	
				alpha_trails = sorted(alpha_trails, key=lambda x: float(x.split('.csv')[0]) ** -1)
				alpha_trails = [trails_file + alpha_trail for alpha_trail in alpha_trails]
				for n, alpha_trail in enumerate(alpha_trails):
					alpha_value = alpha_trail.split('/')[8]
					alpha_value = float(alpha_value.split(".csv")[0])
					alpha_values.append(alpha_value)
					#print alpha_value, 'av'
					#print n,cost_function_i.cost_index
					with open(alpha_trail, 'r') as f:
						trail_reader = csv.reader(f, delimiter=' ', quotechar='|')
						counter = 0
						for row in trail_reader:
							if counter != 0:
								f, a, p, r = row[0].split(',')
								f,a,p,r = cost_function_i.calculate_current_cost([float(f),float(a),float(p),float(r)])
								bootstraps_t.add([f,a,p,r],alpha_value)
								counter +=1
							else:
								counter = 1
						#print len(bootstraps_t.get(alpha_value)), alpha_value,loss_type,alpha_trail
					
			trail_nums = range(20*6*loss_num,20*6*(loss_num+1))
			#print cost_function_i.cost_index
			#alpha_values = [float(a.split(".")[0]) for a in alpha_values]
			metrics = range(3,-1,-1)
			ks = range(0,6)
			for k in ks:
				for x in range(0,4):
					num_vals = range(0,120)
					num = 5
					f, axf 		= P.subplots(5,4,figsize=(16,16),sharex='all',sharey='all',squeeze=True)
					P.suptitle(check(x))
					P.tight_layout()
					sns.set_context("paper")
					P.xlim([0,1])
					count = 0
					for y in num_vals[20*k:20*(k+1)]:
						#cv = crossv.run(Alpha_vals[y],vote_type,loss_type)
						cu = bootstraps_t.get(alpha_values[y])
						cu = np.array(cu)
						#print cu.shape,cu.
						if np.min(cu) < 0:
							print np.min(cu)
						if np.max(cu) > 1:
							print np.max(cu)
						cu = cu[0:100,x]
						#print 'y =',y,'min value = ', np.min(cu),'max value = ', np.max(cu),'mean =', now_mean[y]
						#print "y=",y,"count=",count,"x=",x,"x*y",x*y

						#cost_trail = cost_function_i.get_cost(y)
						sns.kdeplot(cu,ax=axf[count%num][count/num]) # throwing an error at 12?
						axf[count%num][count/num].set_title("C="+"%3.4g" % (float(alpha_values[y])**-1))
						'''
						axf[y%num][y/num].axvline(now_mean[y], ls="--", linewidth=1.5)
						axf[y%num][y/num].axvline(cv[x], ls="--", linewidth=25,color="red")
						axf[y%num][y/num].axvline(now_precent[y][0], ls="-", linewidth=1.25,color="black")
						axf[y%num][y/num].axvline(now_precent[y][1], ls="-", linewidth=1.25,color="black")
						axf[y%num][y/num].set_title("C="+"%3.4g" % (float(Alpha_vals[y])**-1))

						text = '$\mu=%.2f$\n(%.2f,%.2f)'%(float(now_mean[y]), float(now_precent[y][0]),float(now_precent[y][1]))
						props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
						axf[y%num][y/num].text(0.05, 0.95, text, transform=axf[y%num][y/num].transAxes, fontsize=12,
						verticalalignment='top', bbox=props)
						'''
						#print 'Done!'
						#axf[y%5][y/5].set_title("C="+str(Alpha_vals[y]))
						count +=1
					out =  "results/" + str(k+1) + '_'+str(check(x)) + '__' + LT(loss_num) + ".jpg"
					P.savefig(out)
					P.close(out)


if __name__ == "__main__":
	#write()
	cost_function_graphs()
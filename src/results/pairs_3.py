import csv
import numpy as np
import matplotlib.pyplot as P
import os
from os import path
import seaborn as sns
import matplotlib.text as txt
import pdb
from cost_function import cost_function
from pylab import *
import crossv


'''
Calculate the pairs for all of them:
graph a pair of 5 for each (cost type; measure_type,loss_type)
normalize by max cost for the other one

'''


def get_values(loss_type_input):
	loss_types = [loss_type_input]
	file_writer = open(loss_types[0] + '_SR.csv','w')
	file_csv = csv.writer(file_writer)
	file_csv.writerow(('alpha','mean','cv_mean','loss'))
	#print cost_read
	for loss_type in loss_types:
		dirc = '../../output/Systematic_Review/outputs/' + loss_type 
		files = os.listdir(dirc)
		 
		total_measure = {}
		total_cost = {}

		alpha_values = []
		for path in files:  # each file with an alpha value
			alpha = path.split('.csv')[0]
			# print counter%5,alpha
			path = dirc + '/' + path
			#path = file + path
			open_file = open(path, 'r')
			#print "opening_file:", path
			reader = csv.reader(open_file, delimiter=' ', quotechar='|')
			breaks = 0
			measures = []
			costs = []
			max_cost = []

			for read in reader:
				if breaks == 0:
					breaks = +1
				else:
					#print read[0]
					measure,cost = [
						float(x) for x in read[0].split(',')]
					measures.append(measure)
					costs.append(cost)
					
			cv = crossv.get_SR(float(alpha), loss_type)
			measures = np.array(measures)
			costs = np.array(costs)

			alpha = float(alpha)
			alpha_values.append(alpha)
			#alpha = round(alpha)
			total_measure[alpha] = measures
			#print "Measure:", measure
			total_cost[alpha]= costs
			max_cost.append(np.sum(costs))
			#print 'Alpha: ',alpha
			#print cv, alpha, loss_type
			write = [alpha, np.mean(measures), cv, np.sum(costs)]
			file_csv.writerow(write)
			file_csv.writerow(write)


	print max(max_cost)
	return total_measure, total_cost,max(max_cost)

def get_pairs(Measure):  # Cost type is still needed for all
	pairs = []
	#print Measure
	for measure_1 in Measure:
		#print measure_1
		mean_1 =  float('%.2f'% measure_1[1])
		exp_1 = float('%.2f'% measure_1[2])
		for measure_2 in Measure:
			if measure_1 != measure_2:
				mean_2 =  float('%.2f'% measure_2[1])
				exp_2 = float('%.2f' % measure_2[2])
				if (mean_1 > mean_2) and (exp_1 > exp_2):
					# print "%3.4g" % measure_1[0], "%3.4g" % measure_2[0]
					#print exp_1,exp_2
					pairs.append(
						[measure_1[0],measure_2[0],exp_1-exp_2])

	pairs=sorted(pairs, key = lambda x: x[2], reverse=True)
	#print len(pairs)
	return pairs[:5]
def get_num(x): 
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


def plot(pairs, Measures, Costs, loss_type,max_cost):
	f, axf = P.subplots(
	 5, 4, figsize = (16, 16), sharex = 'col', sharey = 'col', squeeze = True)
	P.tight_layout()

	for n,pair in list(enumerate(pairs)):
		C_1=float(pair[0])
		#print C_1, type(Measures)
		C_2=float(pair[1])
		cv_first = crossv.get_SR(C_1,loss_type)
		cv_second = crossv.get_SR(C_2,loss_type)
		first_measure=Measures[C_1]
		#print "typefirst measure:",first_measure
		first_cost = Costs[C_1]
		print "FIRST COST:", type(first_cost)
		second_measure=Measures[C_2]
		second_cost = Costs[C_2]
		C_1 = '%3.4g' % (C_1**-1)
		C_2 = '%3.4g' % (C_2**-1)
		lower_percentile_first = np.percentile(first_measure,25)
		upper_percentile_first = np.percentile(first_measure,75)
		lower_percentile_second = np.percentile(second_measure,25)
		upper_percentile_second = np.percentile(second_measure,75)
		axf[n][
		0].get_yaxis().set_visible(False)
		axf[n][
		1].get_yaxis().set_visible(False)
		axf[n][
		2].get_yaxis().set_visible(False)
		axf[n][
		3].get_yaxis().set_visible(False)

		sns.kdeplot(first_measure, kernel='cos', ax=axf[n][0])
		sns.kdeplot(second_measure, kernel='cos', ax=axf[n][1])
		sns.kdeplot(
			first_cost/max_cost, ax=axf[n][2],label="Cost")
		sns.kdeplot(
			second_cost/max_cost, ax=axf[n][3],label="Cost")
		axf[n][
			0].axvline(np.mean(first_measure), ls="--", linewidth=1.5)
		axf[n][
			1].axvline(np.mean(second_measure), ls="--", linewidth=1.5)
		axf[n][0].axvline( # NEED TO FIX THIS 
			cv_first, ls="--", linewidth=1.25, color="red")
		axf[n][1].axvline(
			cv_second, ls="--", linewidth=1.25, color="red")
		axf[n][0].axvline(
			lower_percentile_first, ls="-", linewidth=1.25, color="black")
		axf[n][1].axvline(
			lower_percentile_second, ls="-", linewidth=1.25, color="black")
		axf[n][0].axvline(
			upper_percentile_first, ls="-", linewidth=1.25, color="black")
		axf[n][1].axvline(
			upper_percentile_second, ls="-", linewidth=1.25, color="black")
		axf[n][0].set_title(
			"C=" + C_1)
		axf[n][1].set_title(
			"C=" + C_2)
		text = '$\hat{\mu}=%.2f$,$\mu=%.2f$\n(%.2f,%.2f)\n $\ \mathcal{L}_{e}=$%.2f' % (cv_first,
				float(np.mean(first_measure)), lower_percentile_first, upper_percentile_first, np.sum(first_cost/max_cost))
		props = dict(
				boxstyle='round', facecolor='wheat', alpha=0.5)
		axf[n][0].text(0.95, 0.95, text, transform=axf[n][0].transAxes, fontsize=10,
							   verticalalignment='top', horizontalalignment='right', bbox=props)
		text = '$\hat{\mu}=%.2f$,$\mu=%.2f$\n(%.2f,%.2f)\n $\ \mathcal{L}_{e}=$%.2f' % (cv_second,
				float(np.mean(second_measure)), lower_percentile_second, upper_percentile_second, np.sum(second_cost/max_cost)) 
		props = dict(
				boxstyle='round', facecolor='wheat', alpha=0.5)
		axf[n][1].text(0.95, 0.95, text, transform=axf[n][1].transAxes, fontsize=10,
							   verticalalignment='top', horizontalalignment='right', bbox=props)
		legend()
	out = "test_results_SR/" + loss_type  + ".jpg"
	print out
	P.savefig(out)
	P.close(out)



Measures_hinge, Cost_hinge,max_hinge=get_values('hinge')
Measures_log, Cost_log,max_log=get_values('log')
hinge_file=open(r'hinge_SR.csv', 'r')
log_file=open(r'log_SR.csv', 'r')
hinge_reader=csv.reader(hinge_file, delimiter = ',', quotechar = '|')
log_reader=csv.reader(log_file, delimiter = ',', quotechar = '|')


breaker=0

Hinge = []
Log = []

for hinge in hinge_reader:
	if breaker == 0:
		breaker += 1
		continue
	else:
		C=float(hinge[0])
		mean=float(hinge[2])
		loss=float(hinge[3])
		Hinge.append([C,mean,loss])
breaker=0
for log in log_reader:
	if breaker == 0:
		breaker += 1
		continue
	else:
		C=float(log[0])
		mean=float(log[2])
		loss=float(log[3])
		Log.append([C,mean,loss])
#print Hinge[0]
pairs_hinge=get_pairs(Hinge)
pairs_log=get_pairs(Log)
plot(pairs_hinge, Measures_hinge, Cost_hinge, 'hinge',max_hinge)
plot(pairs_log, Measures_log, Cost_log, 'log',max_log)

				
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
import sys

'''
Calculate the pairs for all of them:
graph a pair of 5 for each (cost type; measure_type,loss_type)
normalize by max cost for the other one

'''


def get_values(loss_type_input):
    loss_types = [loss_type_input]
    vote_type = "MAX"
    cost_function_i = cost_function()
    file_writer = open(loss_types[0] + '.csv','w')
    file_csv = csv.writer(file_writer)
    file_csv.writerow(('alpha', 'test', 'mean','cv_mean','lower_percentile','upper_percentile','exp','step','linear'))
    file_cost_open =  open(loss_types[0] + '_cost.csv','r')
    o = csv.reader(file_cost_open, delimiter=',', quotechar='|')
    cost_read= [(row[0],row[1])for row in o]
    #print cost_read
    cost_function_i.add_min_max(cost_read)
    for loss_type in loss_types:
        dirc = '../../output/irony/CL/' + loss_type + '/' + vote_type + '/'
        files = os.listdir(dirc)
        files.remove('results')
        files = sorted(files, key=lambda x: int(x.split('_')[1]))
        files = files[:6]
                
        f1_total = {}
        accuracy_total = {}
        precision_total = {}
        recall_total = {}

        f1_total_cost = {}
        accuracy_total_cost = {}
        precision_total_cost = {}
        recall_total_cost = {}

        max_cost_F1 = []
        max_cost_accuracy = []
        max_cost_precision = []
        max_cost_recall = []


        alpha_values = []
        # file = trails_1
        for file_num, file in list(enumerate(files, start=1)):
            #print file.split('_')[1]
            if int(file.split('_')[1]) < 3:
                #print file
                continue
            file = dirc + file + "/"
            alpha_file = os.listdir(file)
            file_t = sorted(
                alpha_file, key=lambda x: float(x.split('.csv')[0]) ** -1)
            counter = 0

            for path in file_t:  # each file with an alpha value
                alpha = path.split('.csv')[0]
                # print counter%5,alpha
                path = file + path
                open_file = open(path, 'r')
                reader = csv.reader(open_file, delimiter=' ', quotechar='|')
                breaks = 0
                f1_alpha = []
                accuracy_alpha = []
                precision_alpha = []
                recall_alpha = []
                f1_alpha_cost = []
                accuracy_alpha_cost = []
                precision_alpha_cost = []
                recall_alpha_cost = []

                for read in reader:
                    if breaks == 0:
                        breaks = +1
                    else:
                        f1, accuracy, precision, recall = [
                            float(x) for x in read[0].split(',')]
                        f1_alpha.append(f1)
                        accuracy_alpha.append(accuracy)
                        precision_alpha.append(precision)
                        recall_alpha.append(recall)
                        f1_cost, accuracy_cost, precision_cost, recall_cost = cost_function_i.calculate_current_cost(
                            [f1, accuracy, precision, recall])
                        f1_alpha_cost.append(f1_cost)
                        accuracy_alpha_cost.append(accuracy_cost)
                        precision_alpha_cost.append(precision_cost)
                        recall_alpha_cost.append(recall_cost)

                cv = crossv.get_irony(float(alpha), vote_type, loss_type)
                f1_alpha = np.array(f1_alpha)
                accuracy_alpha = np.array(accuracy_alpha)
                recall_alpha = np.array(recall_alpha)
                precision_alpha = np.array(precision_alpha)
                f1_alpha_cost = np.array(f1_alpha_cost)
                #print np.sum(f1_alpha_cost[:,0]),f1_alpha_cost[:,0].shape
                accuracy_alpha_cost = np.array(accuracy_alpha_cost)
                precision_alpha_cost = np.array(precision_alpha_cost)
                recall_alpha_cost = np.array(recall_alpha_cost)

                alpha = float(alpha)
                alpha_values.append(alpha)
                #alpha = round(alpha)
                f1_total[alpha] = f1_alpha
                accuracy_total[alpha] = accuracy_alpha
                precision_total[alpha] = precision_alpha
                recall_total[alpha] = recall_alpha

                max_cost_F1.append(np.sum(f1_alpha_cost,axis=0))
                max_cost_accuracy.append(np.sum(accuracy_alpha_cost,axis=0))
                max_cost_precision.append(np.sum(precision_alpha_cost,axis=0))
                max_cost_recall.append(np.sum(recall_alpha_cost,axis=0))

                f1_total_cost[alpha]=f1_alpha_cost
                accuracy_total_cost[alpha]=accuracy_alpha_cost
                precision_total_cost[alpha]=precision_alpha_cost
                recall_total_cost[alpha]=recall_alpha_cost
                #print 'Alpha: ',alpha
                write = [alpha, 'F1', np.mean(f1_alpha), cv[0], np.percentile(f1_alpha, 25), np.percentile(f1_alpha, 75), np.sum(f1_alpha_cost[:,0]), np.sum(f1_alpha_cost[:,1]), np.sum(f1_alpha_cost[:,2])]
                file_csv.writerow(write)
                write = [alpha, 'accuracy', np.mean(accuracy_alpha), cv[1], np.percentile(accuracy_alpha, 25), np.percentile(accuracy_alpha, 75), np.sum(accuracy_alpha_cost[:,0]), np.sum(accuracy_alpha_cost[:,1]), np.sum(accuracy_alpha_cost[:,2])]
                file_csv.writerow(write)
                write = [alpha, 'precision', np.mean(precision_alpha), cv[2], np.percentile(precision_alpha, 25), np.percentile(precision_alpha, 75), np.sum(precision_alpha_cost[:,0]), np.sum(precision_alpha_cost[:,1]), np.sum(precision_alpha_cost[:,2])]
                file_csv.writerow(write)
                write = [alpha, 'recall', np.mean(recall_alpha), cv[3], np.percentile(recall_alpha, 25), np.percentile(recall_alpha, 75), np.sum(recall_alpha_cost[:,0]), np.sum(recall_alpha_cost[:,1]), np.sum(recall_alpha_cost[:,2])]
                file_csv.writerow(write)

    
    #print loss_type_input, 'MAX:', f1_sum #, np.max(accuracy_total_cost[:,0].flatten()),np.max(precision_total_cost[:,0].flatten()),np.max(recall_total_cost[:,0].flatten())
    Measures=[f1_total, accuracy_total, precision_total, recall_total]
    Cost=[f1_total_cost, accuracy_total_cost,
        precision_total_cost, recall_total_cost]
    max_ret = []
    for c in Cost:
        max1 = [np.sum(np.array(c[key][:,0])) for key in c]
        max1 = np.max(max1)
        #print 'MAX', type(max1)
        max_ret.append(float(max1))
    max_ret = [np.max(np.array(max_cost_F1),axis=0),np.max(np.array(max_cost_accuracy),axis=0),np.max(np.array(max_cost_precision),axis=0),np.max(np.array(max_cost_recall),axis=0)]
    return Measures, Cost, max_ret

def get_pairs(Measure):  # Cost type is still needed for all
    pairs = []
    for measure_1 in Measure:
        mean_1 =  float('%.2f'% measure_1[1])
        exp_1 = float('%.2f'% measure_1[3])
        for measure_2 in Measure:
            if measure_1 != measure_2:
                mean_2 =  float('%.2f'% measure_2[1])
                exp_2 = float('%.2f' % measure_2[3])
                if (mean_1 > mean_2) and (exp_1 > exp_2):
                    # print "%3.4g" % measure_1[0], "%3.4g" % measure_2[0]
                    #print exp_1,exp_2
                    pairs.append(
                        [measure_1[0],measure_2[0], mean_1-mean_2])

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


def plot(pairs, Measures, Costs, num,loss_type,max_cost):
    f, axf = P.subplots(
    5, 4, figsize = (16, 16), sharex = 'col', sharey = 'col', squeeze = True)
    P.tight_layout()
    for x in range(0,5):
        for y in range(2,4):
            axf[x][y].set_xlim([0,1])
    for n,pair in list(enumerate(pairs)):
        C_1=float(pair[0])
        #print C_1, type(Measures)
        C_2=float(pair[1])  
        cv_first = crossv.get_irony(C_1, 'MAX', loss_type)
        cv_second = crossv.get_irony(C_2, 'MAX', loss_type)
        first_measure=Measures[num][C_1]
        first_cost = Costs[num][C_1]
        second_measure=Measures[num][C_2]
        second_cost = Costs[num][C_2]
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
            first_cost[:,2], gridsize=50, kernel='cos', ax=axf[n][2],label="linear")
        sns.kdeplot(
            second_cost[:,2], gridsize=50, kernel='cos', ax=axf[n][3],label="linear")
        axf[n][
            0].axvline(np.mean(first_measure), ls="--", linewidth=1.5)
        axf[n][
            1].axvline(np.mean(second_measure), ls="--", linewidth=1.5)
        axf[n][0].axvline( # NEED TO FIX THIS 
            cv_first[num], ls="--", linewidth=1.25, color="red")
        axf[n][1].axvline(
            cv_second[num], ls="--", linewidth=1.25, color="red")
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
        text = '$\hat{\mu}=%.2f$,$\mu=%.2f$\n(%.2f,%.2f)\n $\ \mathcal{L}_{e}=$%.2f' % (cv_first[num],
                float(np.mean(first_measure)), lower_percentile_first, upper_percentile_first, np.sum(first_cost[:,2]/max_cost[num][2]))
        props = dict(
                boxstyle='round', facecolor='wheat', alpha=0.5)
        axf[n][0].text(0.95, 0.95, text, transform=axf[n][0].transAxes, fontsize=10,
                               verticalalignment='top', horizontalalignment='right', bbox=props)
        text = '$\hat{\mu}=%.2f$,$\mu=%.2f$\n(%.2f,%.2f)\n $\ \mathcal{L}_{e}=$%.2f' % (cv_second[num],
                float(np.mean(second_measure)), lower_percentile_second, upper_percentile_second, np.sum(second_cost[:,2]/max_cost[num][2])) 
        props = dict(
                boxstyle='round', facecolor='wheat', alpha=0.5)
        axf[n][1].text(0.95, 0.95, text, transform=axf[n][1].transAxes, fontsize=10,
                               verticalalignment='top', horizontalalignment='right', bbox=props)
        legend()
    out = "test/" + loss_type + '_' + get_num(num) + ".jpg"
    print out
    P.savefig(out)
    P.close(out)


hinge_file=open(r'hinge.csv', 'r')
log_file=open(r'log.csv', 'r')
hinge_reader=csv.reader(hinge_file, delimiter = ',', quotechar = '|')
log_reader=csv.reader(log_file, delimiter = ',', quotechar = '|')
Measures_hinge, Cost_hinge, max_cost_hinge =get_values('hinge')
Measures_log, Cost_log, max_cost_log=get_values('log')

breaker=0

F1_hinge=[]
accuracy_hinge=[]
precision_hinge=[]
recall_hinge=[]

F1_log=[]
accuracy_log=[]
precision_log=[]
recall_log=[]

for hinge in hinge_reader:
    if breaker == 0:
        breaker += 1
        continue
    else:
        C=float(hinge[0])
        mean=float(hinge[3])
        exp=float(hinge[6])
        step=float(hinge[7])
        linear=float(hinge[8])

        if hinge[1] == 'F1':
            F1_hinge.append([C, mean, exp, step, linear])
        elif hinge[1] == 'accuracy':
            accuracy_hinge.append([C, mean, exp, step, linear])
        elif hinge[1] == 'precision':
            precision_hinge.append([C, mean, exp, step, linear])
        else:
            recall_hinge.append([C, mean, exp, step, linear])
Hinge=[F1_hinge, accuracy_hinge, precision_hinge, recall_hinge]
breaker=0
for log in log_reader:
    if breaker == 0:
        breaker += 1
        continue
    else:
        C=float(log[0])
        mean=float(log[3])
        exp=float(log[6])
        step=float(log[7])
        linear=float(log[8])

        if log[1] == 'F1':
            F1_log.append([C, mean, exp, step, linear])
        elif log[1] == 'accuracy':
            accuracy_log.append([C, mean, exp, step, linear])
        elif log[1] == 'precision':
            precision_log.append([C, mean, exp, step, linear])
        else:
            recall_log.append([C, mean, exp, step, linear])
Log=[F1_log, accuracy_log, precision_log, recall_log]
for num, loss in list(enumerate(zip(Hinge, Log))):
    pairs_hinge=get_pairs(loss[0])
    pairs_log=get_pairs(loss[1])
    plot(pairs_hinge, Measures_hinge, Cost_hinge, num,'hinge',max_cost_hinge)
    plot(pairs_log, Measures_log, Cost_log, num,'log',max_cost_log)


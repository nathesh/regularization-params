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
For each alpha and type have the 3 costs, cv, lower and higher percentiles, mean;  i.e. 


'''
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


loss_types = ["hinge"]
vote_type = "MAX"
cost_function_i = cost_function()
cost_file = open("cost_total.csv", 'r')
cost_reader = csv.reader(cost_file, delimiter=',', quotechar='|')
file_writer = open(loss_types[0] + '.csv','w')
file_csv = csv.writer(file_writer)
file_csv.writerow(('alpha', 'test', 'mean','cv_mean','lower_percentile','upper_percentile','exp','step','linear'))
for loss_type in loss_types:
    dirc = '../../output/irony/CL/' + loss_type + '/' + vote_type + '/'
    files = os.listdir(dirc)
    files.remove('results')
    files = sorted(files, key=lambda x: int(x.split('_')[1]))
    files = files[:6]
    cost_read = cost_reader.next()
    cost_read = cost_read[2:]
    cost_read = [float(cr) for cr in cost_read]
    cost_read = [(cost_read[x], cost_read[x + 1])
                 for x in range(0, len(cost_read), 2)]
    cost_function_i.add_min_max(cost_read)
    f1_total = []
    accuracy_total = []
    precision_total = []
    recall_total = []

    f1_total_cost = []
    accuracy_total_cost = []
    precision_total_cost = []
    recall_total_cost = []

    alpha_values = []
    for file_num, file in list(enumerate(files, start=1)):  # file = trails_1
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

            cv = crossv.run(float(alpha), vote_type, loss_type)
            f1_alpha = np.array(f1_alpha)
            accuracy_alpha = np.array(accuracy_alpha)
            recall_alpha = np.array(recall_alpha)
            precision_alpha = np.array(precision_alpha)
            f1_alpha_cost = np.array(f1_alpha_cost)
            accuracy_alpha_cost = np.array(accuracy_alpha_cost)
            precision_alpha_cost = np.array(precision_alpha_cost)
            recall_alpha_cost = np.array(recall_alpha_cost)
            
            alpha = float(alpha)
            alpha_values.append(alpha)

            f1_total.append(f1_alpha)
            accuracy_total.append(accuracy_alpha)
            precision_total.append(precision_alpha)
            recall_total.append(recall_alpha)

            f1_total_cost.append(f1_alpha_cost)
            accuracy_total_cost.append(accuracy_alpha_cost)
            precision_total_cost.append(precision_alpha_cost)
            recall_total_cost.append(recall_alpha_cost)

            cv = crossv.run(alpha, vote_type, loss_type)
            '''
            alpha, test, mean,cv_mean,lower_percentile,upper_percentile,exp,step,linear,
            .0001, Recall,.64, .56, .46,.75,1.25,.56,.63
            '''
            write = [alpha, 'F1', np.mean(f1_alpha), cv[0], np.percentile(f1_alpha, 25), np.percentile(f1_alpha, 75), np.sum(f1_alpha_cost[0]), np.sum(f1_alpha_cost[1]), np.sum(f1_alpha_cost[2])]
            file_csv.writerow(write)
            write = [alpha, 'accuracy', np.mean(accuracy_alpha), cv[1], np.percentile(accuracy_alpha, 25), np.percentile(accuracy_alpha, 75), np.sum(accuracy_alpha_cost[0]), np.sum(accuracy_alpha_cost[1]), np.sum(accuracy_alpha_cost[2])]
            file_csv.writerow(write)
            write = [alpha, 'precision', np.mean(precision_alpha), cv[2], np.percentile(precision_alpha, 25), np.percentile(precision_alpha, 75), np.sum(precision_alpha_cost[0]), np.sum(precision_alpha_cost[1]), np.sum(precision_alpha_cost[2])]
            file_csv.writerow(write)
            write = [alpha, 'recall', np.mean(recall_alpha), cv[3], np.percentile(recall_alpha, 25), np.percentile(recall_alpha, 75), np.sum(recall_alpha_cost[0]), np.sum(recall_alpha_cost[1]), np.sum(recall_alpha_cost[2])]
            file_csv.writerow(write)





    #print len(f1_total), len(accuracy_total_cost),type(f1_total),type(f1_total_cost)
    sys.exit(0)
    total = f1_total + accuracy_total + precision_total + recall_total
    #print type(total), len(total)
    total_cost = f1_total_cost + accuracy_total_cost + \
        precision_total_cost + recall_total_cost
    #print type(total_cost[0]), total_cost[0].shape
    #print type(total_cost), type(total_cost[0])
    for x in range(100, 120, 5):
        for y in range(0, 4):
            f, axf = P.subplots(
                5, 2, figsize=(16, 16), sharex='all', sharey='col', squeeze=True)
            P.tight_layout()
            P.xlim([0,1])
            axf[0][0].get_yaxis().set_visible(False)
            #print y 
            upper = x+4
            lower = x
            #print "lower=",lower,"upper=",upper
            while lower <= upper:
                #print "y=",y
                measure_current = total[lower+y*120]
                current_cost = total_cost[y*120+lower]
                current_mean = np.mean(measure_current)
                cv = crossv.run(alpha_values[lower], vote_type, loss_type)
                lower_percentile = np.percentile(measure_current, 25)
                upper_percentile = np.percentile(measure_current, 75)
                axf[lower%5][
                    0].get_yaxis().set_visible(False)
                axf[lower%5][
                    1].get_yaxis().set_visible(False)
                sns.kdeplot(measure_current, kernel='cos', ax=axf[lower%5][0])
                sns.kdeplot(
                    current_cost[:,0], gridsize=50, kernel='cos', ax=axf[lower%5][1],label="exponential")
                sns.kdeplot(
                    current_cost[:,1], gridsize=50, kernel='cos', ax=axf[lower%5][1],label="step")
                sns.kdeplot(
                    current_cost[:,2], gridsize=50, kernel='cos', ax=axf[lower%5][1],label="linear")
                axf[lower%5][
                    0].axvline(current_mean, ls="--", linewidth=1.5)
                axf[lower%5][0].axvline(
                    cv[y], ls="--", linewidth=1.25, color="red")
                axf[lower%5][0].axvline(
                    lower_percentile, ls="-", linewidth=1.25, color="black")
                axf[lower%5][0].axvline(
                    upper_percentile, ls="-", linewidth=1.25, color="black")
                axf[lower%5][0].set_title(
                    "C=" + "%3.4g" % (float(alpha_values[lower]) ** -1))
                text = '$\hat{\mu}=%.2f$,$\mu=%.2f$\n(%.2f,%.2f)\n $\ \mathcal{L}_{e}=$%.2f,$\mathcal{L}_{s}=$%.2f,$\mathcal{L}_{l}=$%.2f\n' % (cv[y],
                                                                                       float(current_mean), lower_percentile, upper_percentile, np.sum(current_cost[0]),np.sum(current_cost[1]),np.sum(current_cost[2]))
                props = dict(
                    boxstyle='round', facecolor='wheat', alpha=0.5)
                axf[lower%5][0].text(0.95, 0.95, text, transform=axf[lower%5][0].transAxes, fontsize=10,
                                   verticalalignment='top', horizontalalignment='right', bbox=props)
                legend()
                lower += 1

            out = 'test_results/' + loss_type + '_' + \
                vote_type + check(y) + str(x) + ".jpg"  
            print out
            P.savefig(out)
            P.close(out)

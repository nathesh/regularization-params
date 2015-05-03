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
import time 

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

loss_types = ["log", "hinge"]
vote_types = ["MAX"]
cost_function_i = cost_function()
cost_file = open("cost.csv", 'r')
cost_reader = csv.reader(cost_file, delimiter=',', quotechar='|')
for loss_type in loss_types:
    for vote_type in vote_types:
        print 'RUNNING', loss_type, vote_type
        dirc = '../../output/irony/CL/' + loss_type + '/' + vote_type + '/'
        files = os.listdir(dirc)
        files.remove('results')
        files_1 = sorted(files, key=lambda x: int(x.split('_')[1]))
        files_1 = files_1[:6]
        # print files_1
        files_1 = list(enumerate(files_1, start=1))
        for results_1, files in files_1:  # set of 20 trails i.e. trails_1
            # this is the cost for 20 bootstrap trails (100)
            cost_read = cost_reader.next()
            cost_read = cost_read[2:]
            cost_read = [float(cr) for cr in cost_read]
            cost_read = [(cost_read[x], cost_read[x + 1])
                         for x in range(0, len(cost_read), 2)]
            #print cost_read
            #read_num = reader.next()
            cost_function_i.add_min_max(cost_read)
            files = dirc + files + "/"
            files_2 = files
            files = os.listdir(files)
            files = sorted(
                files, key=lambda x: float(x.split('.csv')[0]) ** -1)
            fil = list(enumerate(files))
            f, axf = P.subplots(5, 4, sharex=True)
            P.suptitle("F1")
            f1_t = []
            f1_cost_t = []
            acc_t = []
            acc_cost_t = []
            prec_t = []
            prec_cost_t = []
            rec_t = []
            rec_cost_t = []
            Alpha_vals = []
            f1_mean_t = []
            acc_mean_t = []
            prec_mean_t = []
            rec_mean_t = []
            f1_precentile = []
            acc_precentile = []
            prec_precentile = []
            rec_precentile = []

            for nu, fi in fil:  # alpha value
                alpha = fi.split('.csv')[0]
                #n = "output/alpha_" + str(alpha) + ".csv"
                fip = files_2 + "/" + fi
                Alpha_vals.append(float(alpha))
                with open(fip, 'r') as f:  # Open up for the alpha value
                    spamreader = csv.reader(f, delimiter=' ', quotechar='|')
                    f1 = []
                    f1_cost = []
                    accuracy = []
                    accuracy_cost = []
                    precision = []
                    precision_cost = []
                    recall = []
                    recall_cost = []
                    d = 0
                    for row in spamreader:
                        if d != 0:
                            f, a, p, r = row[0].split(',')
                            f1.append(float(f))
                            accuracy.append(float(a))
                            precision.append(float(p))
                            recall.append(float(r))
                            f, a, p, r = cost_function_i.calculate_current_cost(
                                [float(f), float(a), float(p), float(r)])
                            f1_cost.append(f)
                            accuracy_cost.append(a)
                            precision_cost.append(p)
                            recall_cost.append(r)
                        else:
                            d += 1
                    f1 = np.array(f1)
                    f1_cost = np.array(f1_cost)
                    accuracy = np.array(accuracy)
                    accuracy_cost = np.array(accuracy_cost)
                    precision = np.array(precision)
                    precision_cost = np.array(precision_cost)
                    recall = np.array(recall)
                    recall_cost = np.array(recall_cost)
                    f1_t.append(f1)
                    f1_cost_t.append(f1_cost)
                    acc_t.append(accuracy)
                    acc_cost_t.append(accuracy_cost)
                    prec_t.append(precision)
                    prec_cost_t.append(precision_cost)
                    rec_t.append(recall)
                    rec_cost_t.append(recall_cost)
                    f1_mean = np.mean(f1)
                    accuracy_mean = np.mean(accuracy)
                    precision_mean = np.mean(precision)
                    recall_mean = np.mean(recall)
                    f1_mean_t.append(f1_mean)
                    acc_mean_t.append(accuracy_mean)
                    prec_mean_t.append(precision_mean)
                    rec_mean_t.append(recall_mean)
                    f1_precentile.append(
                        (np.percentile(f1, 25), np.percentile(f1, 75)))
                    acc_precentile.append(
                        (np.percentile(accuracy, 25), np.percentile(accuracy, 75)))
                    prec_precentile.append(
                        (np.percentile(precision, 25), np.percentile(precision, 75)))
                    rec_precentile.append(
                        (np.percentile(recall, 25), np.percentile(recall, 75)))
                    # Focus_3

            All = (f1_t, acc_t, prec_t, rec_t)
            All_cost = (f1_cost_t, acc_cost_t, prec_cost_t, rec_cost_t)
            All_mean = (f1_mean_t, acc_mean_t, prec_mean_t, rec_mean_t)
            All_precentiles = (
                f1_precentile, acc_precentile, prec_precentile, rec_precentile)

            for x in range(0, 4):
                #print check(x)
                # pdb.set_trace()
                #f, axf 		= P.subplots(5,4,figsize=(16,16),sharex='all',sharey='all',squeeze=True)
                num_vals = range(0, 20)
                now = All[x]
                now_cost = All_cost[x]
                num = 5
                f, axf = P.subplots(
                    5, 4, figsize=(16, 16), sharex='all', sharey='all', squeeze=True)
                P.suptitle(check(x))
                P.tight_layout()
                sns.set_context("paper")
                P.xlim([0,1])
                now_mean = All_mean[x]
                now_precent = All_precentiles[x]
                for y in num_vals:
                    
                    cu = now[y]
                    cu_cost = now_cost[y]
                    test = np.vstack((cu, cu_cost))
                    #cu = test 
                    #print cu.shape,cu_cost.shape
                    # print 'y =',y,'min value = ', np.min(cu),'max value = ', np.max(cu),'mean =', now_mean[y]
                    # print cu.shape,cu.size
                    if np.min(cu) == 0 :
                        continue
                    if np.min(cu) <= 0 or np.max(cu) > 1:
                        print 'There are negs!'
                    if y == 18 and x == 1 and float(Alpha_vals[y])- 67857 < 5:
                        print cu_cost
                    print 'Alpha: ', (float(Alpha_vals[y]) ** -1), 'Non-zeros: ', np.nonzero(cu_cost)[0].size,"y: ",y, "x :", x
                    cv = crossv.run(Alpha_vals[y], vote_type, loss_type)
                    #cost_trail = cost_function_i.get_cost(y)
                    # throwing an error at 12?

                    if y == 18 and x == 2:
                        print "CU_COST:\n",cu_cost

                    sns.kdeplot(cu, ax=axf[y % num][y / num])
                    sns.kdeplot(cu_cost, kernel='cos',ax=axf[y % num][y / num])
                    axf[y % num][
                        y / num].axvline(now_mean[y], ls="--", linewidth=1.5)
                    axf[y % num][
                        y / num].axvline(cv[x], ls="--", linewidth=1.25, color="red")
                    axf[y % num][
                        y / num].axvline(now_precent[y][0], ls="-", linewidth=1.25, color="black")
                    axf[y % num][
                        y / num].axvline(now_precent[y][1], ls="-", linewidth=1.25, color="black")
                    axf[y % num][
                        y / num].set_title("C=" + "%3.4g" % (float(Alpha_vals[y]) ** -1))
                    text = '$\mu=%.2f$\n(%.2f,%.2f)' % (
                        float(now_mean[y]), float(now_precent[y][0]), float(now_precent[y][1]))
                    props = dict(
                        boxstyle='round', facecolor='wheat', alpha=0.5)
                    axf[y % num][y / num].text(0.05, 0.95, text, transform=axf[y % num][y / num].transAxes, fontsize=12,
                                               verticalalignment='top', bbox=props)
                    # print 'Done!'
                    # axf[y%5][y/5].set_title("C="+str(Alpha_vals[y]))
                out = '../../output/irony/CL/' + loss_type + '/' + vote_type + \
                    '/results/trails_' + \
                    str(results_1) + '_results/' + check(x) + ".jpg"
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

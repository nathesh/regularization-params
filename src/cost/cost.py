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
    with open("cost_total.csv", 'w') as out:
        csv_out = csv.writer(out)
        ##csv_out.writerow(('loss type','trail number','F1 max', 'F1 min', 'Accuracy max', 'Accuracy min','Precision max','Precision min','Recall max','Recall min'))
        for loss_type in loss_types:
            dirc = '../../output/irony/CL/' + loss_type + '/MAX/'
            files = os.listdir(dirc)
            files.remove('results')
            files_1 = sorted(files, key=lambda x: int(x.split('_')[1]))
            files_1 = files_1[:6]
            counter_files = 0
            f1_t = []
            acc_t = []
            prec_t = []
            rec_t = []
            for file in files_1:  # these are the 6 trails (set of 6 trails)
                current_trail = dirc + file + '/'
                current_files = os.listdir(current_trail)
                current_files = sorted(
                    current_files, key=lambda x: float(x.split('.csv')[0]) ** -1)

                # 20 trails in that 6 trails
                for current_file in current_files:
                    alpha = current_file.split('.csv')[0]
                    open_file = current_trail + current_file
                    # this for each trail 20 of these
                    with open(open_file, 'r') as f:
                        spamreader = csv.reader(
                            f, delimiter=' ', quotechar='|')
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
            write = (loss_type, counter_files, np.percentile(f1_t, 75), np.percentile(f1_t, 33), \
                     np.percentile(acc_t, 75), np.percentile(acc_t, 33), np.percentile(prec_t, 75), \
                     np.percentile(prec_t, 33), np.percentile(rec_t, 75), np.percentile(rec_t, 33))
            csv_out.writerow(write)

            print "loss type: ", loss_type, counter_files + 1
            print "\nF1: ", np.max(f1_t),   np.percentile(f1_t, 33)
            print "Accuracy: ", np.max(acc_t),  np.percentile(acc_t, 33)
            print "Precision: ", np.max(prec_t), np.percentile(prec_t, 33)
            print "Recall: ", np.max(rec_t),  np.percentile(rec_t, 33)
            print "\n"


def cost_function_graphs():
    cost_function_i = cost_function()
    with open("test.csv", 'r') as cost_file:
        cost_reader = csv.reader(cost_file, delimiter=',', quotechar='|')
        loss_types = ['log', 'hinge']
        for loss_num, loss_type in enumerate(loss_types):
            alpha_values = []
            path = '../../output/irony/CL/' + loss_type + '/MAX/'
            trails_files = os.listdir(path)
            trails_files.remove('results')
            trails_files = sorted(
                trails_files, key=lambda x: int(x.split('_')[1]))
            trails_files = [path + file + "/" for file in trails_files]
            trails_files = trails_files[:6]
            for trails_file in trails_files:
                # this is the cost for 20 bootstrap trails (100)
                cost_read = cost_reader.next()
                cost_read = cost_read[2:]
                cost_read = [float(cr) for cr in cost_read]
                cost_read = [(cost_read[x], cost_read[x + 1])
                             for x in range(0, len(cost_read), 2)]
                # print cost_read
                cost_function_i.add_min_max(cost_read)
                alpha_trails = os.listdir(trails_file)
                alpha_trails = sorted(
                    alpha_trails, key=lambda x: float(x.split('.csv')[0]) ** -1)
                alpha_trails = [
                    trails_file + alpha_trail for alpha_trail in alpha_trails]
                for n, alpha_trail in enumerate(alpha_trails):
                    alpha_value = alpha_trail.split('/')[8]
                    alpha_values.append(alpha_value)
                    cost_function_i.next()
                    # print n,cost_function_i.cost_index
                    with open(alpha_trail, 'r') as f:
                        trail_reader = csv.reader(
                            f, delimiter=' ', quotechar='|')
                        counter = 0
                        for row in trail_reader:
                            if counter != 0:
                                f, a, p, r = row[0].split(',')
                                cost_function_i.add_values(
                                    [float(f), float(a), float(p), float(r)])
                                cost_function_i.calculate_current_cost([])
                            else:
                                counter = 1
                    cost_function_i.calculate()
            trail_nums = range(20 * 6 * loss_num, 20 * 6 * (loss_num + 1))
            # print cost_function_i.cost_index
            alpha_values = [float(a.split(".csv")[0]) for a in alpha_values]
            metrics = range(3, -1, -1)
            for metric in metrics:
                costs = []
                costs = [cost_function_i.get_cost(
                    trail_num, metric) for trail_num in trail_nums]
                costs = np.array(costs)
                #costs = [cost/100 for cost in costs]
                alpha_values = np.array(alpha_values)
                print len(alpha_values), costs[:, 0].size
                np.set_printoptions(suppress=True)
                P.figure()
                f, axf = P.subplots(squeeze=False)
                P.suptitle(check(metric) + "_" + LT(loss_num))
                P.plot(alpha_values, costs[:, 0], label='exponential')
                P.plot(alpha_values, costs[:, 1], label='step')
                P.plot(
                    alpha_values, costs[:, 2], label='linear', linestyle='--', color='black')
                #P.xticks(range(120), alpha_values[0:10], size='small')
                # gca().xaxis.set_major_formatter(OldScalarFormatter())
                #ax = gca()
                xscale('log')
                ax = gca().xaxis
                ax.set_major_formatter(ScalarFormatter())
                P.legend()
                # P.xlim([0,100])
                #P.ylim([0, .4])
                ax = gca().xaxis

                outp = "pics/" + \
                    str(check(metric)) + '__' + LT(loss_num) + ".jpg"
                P.savefig(outp)
                P.close(outp)
                # P.flush()


if __name__ == "__main__":
    write()
    # cost_function_graphs()

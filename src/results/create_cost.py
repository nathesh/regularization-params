import csv
import os
import numpy as np 
loss_type = 'log'
vote_type = "MAX"

dirc = '../../output/20_newsgroups/ath.v.chs/outputs/' + loss_type 
files = os.listdir(dirc)
#files.remove('results')
f1_total = []
accuracy_total = []
precision_total = []
recall_total = []

alpha_values = []
for path in files:  # each file with an alpha value
    alpha = path.split('.csv')[0]
    # print counter%5,alpha
    path = dirc + '/' + path
    open_file = open(path, 'r')
    reader = csv.reader(open_file, delimiter=' ', quotechar='|')
    breaks = 0
    f1_alpha = []
    accuracy_alpha = []
    precision_alpha = []
    recall_alpha = []

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

    f1_alpha = np.array(f1_alpha)
    accuracy_alpha = np.array(accuracy_alpha)
    recall_alpha = np.array(recall_alpha)
    precision_alpha = np.array(precision_alpha)

    alpha = float(alpha)
    alpha_values.append(alpha)
    #alpha = round(alpha)
    f1_total.append(f1_alpha)
    accuracy_total.append(accuracy_alpha)
    precision_total.append(precision_alpha)
    recall_total.append(recall_alpha)

file_writer = open(loss_type + 'NG_cost.csv','w')
file_csv = csv.writer(file_writer)
file_csv.writerow([np.percentile(np.array(f1_total),25),np.percentile(np.array(f1_total),75)
])
file_csv.writerow((np.percentile(np.array(accuracy_total),25),np.percentile(np.array(accuracy_total),75)
))
file_csv.writerow((np.percentile(np.array(precision_total),25),np.percentile(np.array(precision_total),75)
))
file_csv.writerow((np.percentile(np.array(recall_total),25),np.percentile(np.array(recall_total),75)
))


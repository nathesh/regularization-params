import csv
import os
import numpy as np 
loss_type = 'hinge'
vote_type = "MAX"

dirc = '../../output/irony/CL/' + loss_type + '/' + vote_type + '/'
files = os.listdir(dirc)
files.remove('results')
files = sorted(files, key=lambda x: int(x.split('_')[1]))
files = files[:6]
f1_total = []
accuracy_total = []
precision_total = []
recall_total = []

alpha_values = []
# file = trails_1
for file_num, file in list(enumerate(files, start=1)):
    print file.split('_')[1]
    if int(file.split('_')[1]) < 3:
        print file
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

file_writer = open(loss_type + '_cost.csv','w')
file_csv = csv.writer(file_writer)
file_csv.writerow([np.percentile(np.array(f1_total),25),np.percentile(np.array(f1_total),75)
])
file_csv.writerow((np.percentile(np.array(accuracy_total),25),np.percentile(np.array(accuracy_total),75)
))
file_csv.writerow((np.percentile(np.array(precision_total),25),np.percentile(np.array(precision_total),75)
))
file_csv.writerow((np.percentile(np.array(recall_total),25),np.percentile(np.array(recall_total),75)
))


import numpy
import os
import csv
import numpy as np
'''
def step():



def linear():


def exponential():
'''


if __name__ == "__main__":
	loss_types = ['log', 'hinge']

	with open("test.csv", 'w') as out:
		csv_out = csv.writer(out)
		##csv_out.writerow(('loss type','trail number','F1 max', 'F1 min', 'Accuracy max', 'Accuracy min','Precision max','Precision min','Recall max','Recall min'))
			
		for loss_type in loss_types:
			dirc = '../../output/irony/CL/' + loss_type + '/MAX/'
			files = os.listdir(dirc)
			files.remove('results')
			files_1 = sorted(files, key=lambda x: int(x.split('_')[1]))
			files_1 = files_1[:6]
			counter_files = 0
			for file in files_1:  # these are the 6 trails (set of 6 trails)
				current_trail = dirc + file + '/'
				current_files = os.listdir(current_trail)
				current_files = sorted(
				    current_files, key=lambda x: float(x.split('.csv')[0]) ** -1)
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


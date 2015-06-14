    #!/usr/local/bin/python
# all the imports
from sklearn.datasets import fetch_20newsgroups as ngs
from sklearn import cross_validation as cv
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import csv
import sys
import os
#import threading
import re
#import easygui
import sqlite3
from data import text_label
import irony
import twenty_newsgroup
import Systematic_Review as SR 
from multiprocessing import Pool,Lock,Process   

def trails(alpha):
    data_set = SR.get_all()
    data = np.array(data_set.data)
    target_vals = np.array(data_set.target)
    losses = ['log','hinge']
    bs = cv.Bootstrap(  len(data), n_iter=100)
    vectorizer = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 2), min_df=3, max_features=50000)
    for loss in losses:
        scores = []
        for train_index, test_index in bs:
            train = data[train_index]
            test = data[test_index]
            train_target_vals = target_vals[train_index]
            test_traget_vals = target_vals[test_index]
            fit = vectorizer.fit(train)
            vector_train = fit.transform(train)
            vector_test = fit.transform(test)
            model_1 =  linear_model.SGDClassifier(loss=loss, alpha=alpha)
            model_1.fit(vector_train, train_target_vals)
            predict = model_1.predict(vector_test)
            measure = metrics.fbeta_score(test_traget_vals,predict,beta=10)
            cost_eval = metrics.confusion_matrix(test_traget_vals,predict)
            FP = cost_eval[1][0]
            FN = cost_eval[0][1]
            cost = 10*FN + FP
            scores.append((measure,cost))
        # Lock right here
        #lock.acquire()
        print "Alpha:", alpha 
        name = '../../../output/Systematic_Review/output/' + loss + "/" + str(alpha) + '.csv'
        with open(name, 'w') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(('measure','cost'))
            for row in scores:
                csv_out.writerow(row)
        #lock.release()
    # Unlock




if __name__ == "__main__":  # inputs -> (dataset,model used)
    # setting up bootstrap
    # getting a deprecationwarning => need to look into it
    '''
    dataset_use = easygui.buttonbox(
        'Click on the dataset to use.', 'Dataset', ('20_newsgroups', 'Irony'))
    if dataset_use == '20_newsgroups':
        dataset_use = easygui.buttonbox(
            'Click on the type of binary classification.', '20_newsgroups', ('Atheism vs. All', 'Atheism vs. Christianity'))
    '''
    alpha_val_range = []
    for i in range(5, 10):
        alpha_val_range.append((10 ** -i, 10 ** -(i + 1), i - 2))
    #multiprocessing.Array(ctypes.c_double, 10*10)
    pool = Pool(5)
    #lock = Lock()
    for x, y, z  in alpha_val_range:
        alpha_vals = np.linspace(y, x, 20)  # trail 8
        print 'z', z    
        print alpha_vals[:10]
        break
        #pool.map(trails,alpha_vals) 
    print "Done?"


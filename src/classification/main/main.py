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

def data(input, type_vote=None):  # return the data
    #print type_vote,input
    if input == 'Atheism vs. Christianity':
        cats = ['alt.atheism', 'Christianity']
        # I got all (train and test)?? remove stuff
        ngs_1 = ngs(subset='all', categories=cats, remove=(
            'headers', 'footers', 'quotes'))
        return ngs_1
    elif input == 'Atheism vs. All':
        return twenty_newsgroup.run()
    elif input == 'Irony-all':
        return irony.get_all(type_vote)
    elif input == 'Irony-CL':
        #print "Why no get here?"
        test = irony.get_conservatives_liberal(type_vote)
        return test



def clean(input):  # return data, target, vectorizer and length
    # covert ngs.data into a numpy array and getting the length
    length = len(input.data)
    #print (input.data)
    data_set = np.array(input.data)
    target_vals = np.array(input.target)
    print data_set.shape, target_vals.shape
    # using the SGD model
    vectorizer = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 2), min_df=3, max_features=50000)
    return (data_set, target_vals, vectorizer, length)


def model(loss, alpha):
    return linear_model.SGDClassifier(loss=loss, alpha=alpha)


def trails_bs(data, target_vals, vectorizer, bs, ml, alpha, dataset_use, doc):
    print alpha
    scores = []
    for train_index, test_index in bs:
        train = data[train_index]
        # print type(train[0])

        #train = train.encode('latin1')
        #train = train.encode('latin-1')
        test = data[test_index]
        train_target_vals = target_vals[train_index]
        test_traget_vals = target_vals[test_index]
        fit = vectorizer.fit(train)
        vector_train = fit.transform(train)
        vector_test = fit.transform(test)
        # vector_train = vectorizer.fit_transform(train)
        # vector_test = vectorizer.transform(test)

        model_1 = model(ml, alpha)
        model_1.fit(vector_train, train_target_vals)
        predict = model_1.predict(vector_test)
        # The metrics that I am going to use are accuracy,percision, F1 and
        # recall
        f1 = metrics.f1_score(test_traget_vals, predict)
        accuracy = metrics.accuracy_score(test_traget_vals, predict)
        precision = metrics.precision_score(test_traget_vals, predict)
        recall = metrics.recall_score(test_traget_vals, predict)
        measures = (f1, accuracy, precision, recall)
        scores.append(measures)
    output(scores, alpha, dataset_use, doc)


def trails(data, target_vals, vectorizer, bs, ml, alpha_vals, dataset_use, doc):
    if alpha_vals is None:
        alpha_vals = []
        for c in range(3, -4, -1):
            a = 10 ** c
            alpha_vals.append(a)

    for alpha in alpha_vals:
        trails_bs(
            data, target_vals, vectorizer, bs, ml, alpha, dataset_use, doc)


def output(scores, alpha, dataset, doc):
    print "in output"
    name = ''
    if dataset == 'Irony':
        name = doc + str(alpha) + '.csv'
    else:
        name = '../../../output/20_newsgroups/ath.v.all/outputs/' + \
            str(alpha) + '.csv'  # need to change this

    with open(name, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(('f1', 'accuracy', 'precision', 'recall'))
        for row in scores:
            csv_out.writerow(row)

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
    types_vote = ['MAJORITY']
    types_model = ['log', 'hinge']
    for type_vote in types_vote:
        data = data('Atheism vs. All', type_vote)  # input
        data, target_vals, vectorizer, length = clean(data)
        bs = cv.Bootstrap(length, n_iter=100)
        alpha_val_range = []
        for i in range(5, 10):
            alpha_val_range.append((10 ** -i, 10 ** -(i + 1), i - 2))

        for x, y, z in alpha_val_range:
            alpha_vals = np.linspace(y, x, 20)  # trail 8
            for type_model in types_model:
                doc = '../../../output//20_newsgroups/ath.v.all/outputs/'
                trails(
                    data, target_vals, vectorizer, bs, type_model, alpha_vals, 'Atheism vs. All', doc)
        print "Done?"

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
import threading


def data(input):  # return the data
    if input == 0:
        cats = ['alt.atheism','sci.space']
        # I got all (train and test)?? remove stuff
        ngs_1 = ngs(subset='all', categories=cats, remove=(
            'headers', 'footers', 'quotes'))
        return ngs_1


def clean(input):  # return data, target, vectorizer and length
    # covert ngs.data into a numpy array and getting the length
    length = len(input.data)
    data = np.array(input.data)
    target_vals = np.array(input.target)
    # using the SGD model
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    return (data, target_vals, vectorizer, length)


def model(type, alpha):
    if type == 0:
        return linear_model.SGDClassifier(alpha=alpha)


def trails_bs(data, target_vals, vectorizer, bs, ml, alpha):
    print alpha
    scores = []
    for train_index, test_index in bs:
        train = data[train_index]
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
    output(scores, alpha)


def trails(data, target_vals, vectorizer, bs, ml, alpha_vals):
    if alpha_vals is None:
        alpha_vals = []
        for c in range(3, -4, -1):
            a = 10 ** c
            alpha_vals.append(a)

    for alpha in alpha_vals:
        trails_bs(data, target_vals, vectorizer, bs, ml, alpha)


def output(scores, alpha):
    print "in output"
    name = '../Athvs.Sci/output_1/alpha_' + str(alpha) + '.csv'
    with open(name, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(('f1', 'accuracy', 'precision', 'recall'))
        for row in scores:
            csv_out.writerow(row)

if __name__ == "__main__":  # inputs -> (dataset,model used)
    # setting up bootstrap
    # getting a deprecationwarning => need to look into it
    data = data(0)  # input
    data, target_vals, vectorizer, length = clean(data)
    bs = cv.Bootstrap(length, n_iter=100)
    alpha_vals = np.linspace(.000001, .01, 20)  # input
    #alpha_vals = [.0001]
    trails(data, target_vals, vectorizer, bs, 0, alpha_vals)
    print "Done?"

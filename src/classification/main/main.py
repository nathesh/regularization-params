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


class ip:
    data = []
    kv = {}
    target = []

    def add(self, dt, target):
        self.data.append(dt)
        self.target.append(target)

    def add_all(self, data_input, target_input):
        self.data = data_input
        self.target = target_input

    def strip_newsgroup_header(self, text):
        """
        Given text in "news" format, strip the headers, by removing everything
        before the first blank line.
        """
        _before, _blankline, after = text.partition('\n\n')
        return after

    def strip_newsgroup_quoting(self, text):
        """
        Given text in "news" format, strip lines beginning with the quote
        characters > or |, plus lines that often introduce a quoted section
        (for example, because they contain the string 'writes:'.)
        """
        _QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                               r'|^In article|^Quoted from|^\||^>)')
        good_lines = [line for line in text.split('\n')
                      if not _QUOTE_RE.search(line)]
        return '\n'.join(good_lines)

    def strip_newsgroup_footer(self, text):
        """
        Given text in "news" format, attempt to remove a signature block.
        As a rough heuristic, we assume that signatures are set apart by either
        a blank line or a line made of hyphens, and that it is the last such line
        in the file (disregarding blank lines at the end).
        """
        lines = text.strip().split('\n')
        for line_num in range(len(lines) - 1, -1, -1):
            line = lines[line_num]
            if line.strip().strip('-') == '':
                break

        if line_num > 0:
            return '\n'.join(lines[:line_num])
        else:
            return text


def data(input):  # return the data
    if input == 'Atheism vs. Christianity':
        cats = ['alt.atheism', 'sci.space']
        # I got all (train and test)?? remove stuff
        ngs_1 = ngs(subset='all', categories=cats, remove=(
            'headers', 'footers', 'quotes'))
        return ngs_1
    elif input == 'Atheism vs. All':
        #f = open("../../data/20_newsgroups/alt.atheism/49960",'r')
        s = '../../../data/newsgroups/20_newsgroups'
        files = os.listdir(s)
        files.sort()
        fi = files.pop(0)
        s_c = s + fi + '/'
        paths = os.listdir(s_c)
        dt = ip()
        for path in paths:

            fin = open(s_c + path, 'r')
            data = fin.read()  # need to change the where the is stored
            data = (data.decode('latin1'))

            data = dt.strip_newsgroup_header(data)
            data = dt.strip_newsgroup_footer(data)
            data = dt.strip_newsgroup_quoting(data)
            dt.add(data, 0)  # data and target
            target = 0
        #num = 0
        for f in files:
            s_c = s + f + '/'
            paths = os.listdir(s_c)
            for path in paths:
                fin = open(s_c + path, 'r')
                data = fin.read()  # need to c
                data = (data.decode('latin1'))
                data = dt.strip_newsgroup_header(data)
                data = dt.strip_newsgroup_footer(data)
                data = dt.strip_newsgroup_quoting(data)
                dt.add(data, 1)
                target = 1
            #num += 1
        return dt
    elif input == 'Irony':
        db_path = '../../../data/irony/ironate.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT text,label FROM irony_commentsegment,irony_label WHERE irony_label.segment_id = irony_commentsegment.id GROUP BY irony_commentsegment.id')
        text_labels = cursor.fetchall()
        text = map(lambda x: x[0], text_labels)
        labels = map(lambda x: x[1], text_labels)
        data = ip()
        data.add_all(text, labels)
        return data


def clean(input):  # return data, target, vectorizer and length
    # covert ngs.data into a numpy array and getting the length
    length = len(input.data)
    #print (input.data)
    data_set = np.array(input.data)
    target_vals = np.array(input.target)
    # using the SGD model
    vectorizer = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 2), min_df=3, max_features=50000)
    return (data_set, target_vals, vectorizer, length)


def model(type, alpha):
    if type == 0:
        return linear_model.SGDClassifier(alpha=alpha)


def trails_bs(data, target_vals, vectorizer, bs, ml, alpha, dataset_use):
    print alpha
    scores = []
    for train_index, test_index in bs:
        train = data[train_index]
        #print type(train[0])

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
    output(scores, alpha, dataset_use)


def trails(data, target_vals, vectorizer, bs, ml, alpha_vals, dataset_use):
    if alpha_vals is None:
        alpha_vals = []
        for c in range(3, -4, -1):
            a = 10 ** c
            alpha_vals.append(a)

    for alpha in alpha_vals:
        trails_bs(data, target_vals, vectorizer, bs, ml, alpha, dataset_use)


def output(scores, alpha, dataset):
    print "in output"
    name = ''
    if dataset == 'Irony':
        name = '../../../output/irony/trails_3/' + str(alpha) + '.csv'
    else:
        name = '../../../output/20_newsgroups/ath.v.all/output_1/alpha_' + \
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
    data = data('Irony')  # input
    data, target_vals, vectorizer, length = clean(data)
    bs = cv.Bootstrap(length, n_iter=100)
    alpha_vals = np.linspace(.0000001, .000001, 20)  # input
    #alpha_vals = [.0001]

    trails(data, target_vals, vectorizer, bs, 0, alpha_vals, 'Irony')
    print "Done?"

''' 
min df: 3 
reduce feature space: 50,000 
create a subprocess
'''

''' Svm -> SGD with hinge loss; L2 norm 
    SGD
'''

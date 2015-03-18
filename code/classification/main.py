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
import threading
import re

'''
NEED TO FIX strip all of them ...
'''


class ip:
    data = []
    kv = {}
    target = []
    def add(self,dt,target):
        self.data.append(dt)
        self.target.append(target)
    def strip_newsgroup_header(self,text):
        """
        Given text in "news" format, strip the headers, by removing everything
        before the first blank line.
        """
        _before, _blankline, after = text.partition('\n\n')
        return after


    def strip_newsgroup_quoting(self,text):
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

    def strip_newsgroup_footer(self,text):
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
    if input == 0:
        cats = ['alt.atheism','sci.space']
        # I got all (train and test)?? remove stuff
        ngs_1 = ngs(subset='all', categories=cats, remove=(
            'headers', 'footers', 'quotes'))
        return ngs_1    
    elif input == 1:
        #f = open("../../data/20_newsgroups/alt.atheism/49960",'r')
        s = '../../data/20_newsgroups/'
        files = os.listdir(s)
        files.sort()
        fi = files.pop(0)
        s_c = s+fi+'/'
        paths = os.listdir(s_c) 
        dt = ip()
        for path in paths:

            fin = open(s_c+path, 'r') 
            data = fin.read() # need to change the where the is stored 
            data = (data.decode('latin1'))
	    
            #data = [dt.strip_newsgroup_header(text) for text in data]
            #data = [dt.strip_newsgroup_footer(text) for text in data]            
            #data = [dt.strip_newsgroup_quoting(text) for text in data]       
            dt.add(data,0) # data and target
            target = 0
        num = 0
        for f in files:
            if num <3:
                s_c = s + f+'/'
                paths= os.listdir(s_c)
                for path in paths:
                    fin = open(s_c+path, 'r') 
                    data = fin.read() # need to c
                    data = (data.decode('latin1'))
                    #data = [dt.strip_newsgroup_header(text) for text in data]
                    #data = [dt.strip_newsgroup_footer(text) for text in data]            
                    #data = [dt.strip_newsgroup_quoting(text) for text in data]
                    dt.add(data,1)
                    target = 1
            num +=1
        return dt

def clean(input):  # return data, target, vectorizer and length
    # covert ngs.data into a numpy array and getting the length
    length = len(input.data)
    #print (input.data)
    data_set = np.array(input.data)
    target_vals = np.array(input.target)
    # using the SGD model
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    return (data_set, target_vals, vectorizer, length)


def model(type, alpha):
    if type == 0:
        return linear_model.SGDClassifier(alpha=alpha)


def trails_bs(data, target_vals, vectorizer, bs, ml, alpha):
    print alpha
    scores = []
    for train_index, test_index in bs:
        train = data[train_index]
        print type(train[0])

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
    name = '../../Athvs.All/output_1/alpha_' + str(alpha) + '.csv'
    with open(name, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(('f1', 'accuracy', 'precision', 'recall'))
        for row in scores:
            csv_out.writerow(row)

if __name__ == "__main__":  # inputs -> (dataset,model used)
    # setting up bootstrap
    # getting a deprecationwarning => need to look into it
    inp = raw_input("Type 0 for athesim vs. religion\nType 1 for athesim vs. all\nThen press Enter...")

    data = data(int(inp))  # input
    data, target_vals, vectorizer, length = clean(data)
    bs = cv.Bootstrap(length, n_iter=10)
    alpha_vals = np.linspace(.000001, .01, 20)  # input
    #alpha_vals = [.0001]
    trails(data, target_vals, vectorizer, bs, 0, alpha_vals)
    print "Done?"


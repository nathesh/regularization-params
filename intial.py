# all the imports
from sklearn.datasets import fetch_20newsgroups as ngs
from sklearn import cross_validation as cv
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import csv
import sys


def data(input):  # return the data
    if input == 0:
        cats = ['alt.atheism', 'talk.religion.misc']
        # I got all (train and test)?? remove stuff
        ngs = ngs(subset='all', categories=cats, remove=(
            'headers', 'footers', 'quotes'))
        return ngs


def clean(input):  # return data, target, vectorizer and length
    # covert ngs.data into a numpy array and getting the length
    length = len(ngs.data)
    data = np.array(ngs.data)
    target_vals = np.array(ngs.target)
    # using the SGD model
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    return (data, target_vals, vectorizer, length)


def model(type, alpha):
    if type == 0:
        return linear_model.SGDClassifier(alpha=alpha)


def trails(data, target_vals, vectorizer, bs, ml):
    for c in range(3, -4, -1):
        alpha = 10 ** c
        scores = []
        for train_index, test_index in bs:
            train = data[train_index]
            test = data[test_index]
            train_target_vals = target_vals[train_index]
            test_traget_vals = target_vals[test_index]
            fit = vectorizer.fit(train)
            vector_train = fit.transform(train)
            vector_test = fit.transform(test)
            #vector_train = vectorizer.fit_transform(train)
            #vector_test = vectorizer.transform(test)

            model = model(m, alpha)
            model.fit(vector_train, train_target_vals)
            predict = model.predict(vector_test)
            # The metrics that I am going to use are accuracy,percision, F1 and
            # recall
            f1 = metrics.f1_score(test_traget_vals, predict)
            accuracy = metrics.accuracy_score(test_traget_vals, predict)
            precision = metrics.precision_score(test_traget_vals, predict)
            recall = metrics.recall_score(test_traget_vals, predict)
            measures = (f1, accuracy, precision, recall)
            scores.append(measures)
            return scores


def output(scores):
    name = 'output/alpha_' + str(alpha) + '.csv'
    with open(name, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(('f1', 'accuracy', 'precision', 'recall'))
        for row in scores:
            csv_out.writerow(row)

if __name__ == "__main__":  # inputs -> (dataset,model used)
    # setting up bootstrap
    # getting a deprecationwarning => need to look into it
    data = data(0)
    data, target_vals, vectorizer, length = clean(data)
    bs = cv.Bootstrap(length, n_iter=100)
    scores = trails(data, target_vals, vectorizer, bs, 0)
    output(scores)


''' TO DO 
1. fix the vectorizer -> DONE
2. debug  - DONE
3. make the plots 
4. Modualize the code -> Done 
5. PEP8 - DONE 
6. Finish the plots 
7. Better cats balanced and higher accuary (varied dataset)
'''

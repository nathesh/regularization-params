# all the imports
from sklearn.datasets import fetch_20newsgroups as ngs
from sklearn import cross_validation as cv
import numpy as np
from sklearn  import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import csv

# Import the data from the 20 news groups; can remove headers and footers
# if need be?
cats = ['alt.atheism', 'talk.religion.misc']
# I got all (train and test)?? remove stuff
ngs = ngs(subset='all', categories=cats, remove=(
    'headers', 'footers', 'quotes'))


# setting up bootstrap
length = len(ngs.data)
# getting a deprecationwarning => need to look into it
bs = cv.Bootstrap(length, n_iter=100)
# using the SGD model
vectorizer = TfidfVectorizer()  # TFIDF vectorizer and need to add the unigram and bigram and stop words 
for c in range(3, -4, -1):
    alpha = 10 ** c
    scores = []
    for train_index, test_index in bs:
    	print type(train_index)
        train = ngs.data[train_index]
        test = ngs.data[test_index]

        vector_train = vectorizer.fit_transform(train)
        vector_test = vectorizer.fit_transform(test)

        sgd = linear_model.SGDClassifier(alpha=alpha)
        sgd.fit(vector_train, train.target)
        predict = sgd.predict(vector_test)
        # The metrics that I am going to use are accuracy,percision, F1 and
        # recall
        f1 = metrics.f1_score(test.target, predict)
        accuracy = metrics.accuracy_score(test.target, predict)
        precision = metrics.precision_score(test.target, predict)
        recall = metrics.recall_score(test.target, predict)
        measures = (f1, accuracy, precision, recall)
        scores.append(measures)
    name = 'alpha_' + str(alpha) + '.csv'
    with open(name, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow('f1', 'accuracy', 'precision', 'recall')
        for row in scores:
            csv_out.writerow(row)

''' TO DO 
1. fix the vectorizer
2. debug 
3. make the plots 
'''




from sklearn.cross_validation import KFold
from sklearn.datasets import fetch_20newsgroups as ngs
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import irony  # need to fix the location of this
import os 

def data_d(dataset, vote):
    return irony.get_conservatives_liberal(vote)


def run(alpha, vote, loss):
    data_s = data_d('Irony-CL', vote)
    data, target_vals, vectorizer, length = clean(data_s)
    scores = []
    kf = KFold(length, n_folds=5)
    for train_index, test_index in kf:
        test = data[test_index]
        train = data[train_index]
        train_target_vals = target_vals[train_index]
        test_traget_vals = target_vals[test_index]
        fit = vectorizer.fit(train)
        vector_train = fit.transform(train)
        vector_test = fit.transform(test)
        model = linear_model.SGDClassifier(loss=loss, alpha=alpha)
        model.fit(vector_train, train_target_vals)
        predict = model.predict(vector_test)
        f1 = metrics.f1_score(test_traget_vals, predict)
        accuracy = metrics.accuracy_score(test_traget_vals, predict)
        precision = metrics.precision_score(test_traget_vals, predict)
        recall = metrics.recall_score(test_traget_vals, predict)
        measures = [f1, accuracy, precision, recall]
        scores.append(measures)
    scores = np.array(scores)  # return F,A,P,R
    return np.mean(scores[:, 0]), np.mean(scores[:, 1]), np.mean(scores[:, 2]), np.mean(scores[:, 3])

def run_NG(alpha, vote, loss):
    cats = ['alt.atheism', 'soc.religion.christian']
        # I got all (train and test)?? remove stuff
    data_s = ngs(subset='all', categories=cats, remove=(
            'headers', 'footers', 'quotes'))

    data, target_vals, vectorizer, length = clean(data_s)
    scores = []
    kf = KFold(length, n_folds=5)
    for train_index, test_index in kf:
        test = data[test_index]
        train = data[train_index]
        train_target_vals = target_vals[train_index]
        test_traget_vals = target_vals[test_index]
        fit = vectorizer.fit(train)
        vector_train = fit.transform(train)
        vector_test = fit.transform(test)
        model = linear_model.SGDClassifier(loss=loss, alpha=alpha)
        model.fit(vector_train, train_target_vals)
        predict = model.predict(vector_test)
        f1 = metrics.f1_score(test_traget_vals, predict)
        accuracy = metrics.accuracy_score(test_traget_vals, predict)
        precision = metrics.precision_score(test_traget_vals, predict)
        recall = metrics.recall_score(test_traget_vals, predict)
        measures = [f1, accuracy, precision, recall]
        scores.append(measures)
    scores = np.array(scores)  # return F,A,P,R
    a = np.asarray([np.mean(scores[:, 0]), np.mean(scores[:, 1]), np.mean(scores[:, 2]), np.mean(scores[:, 3])])
    np.savetxt("cv/" + loss + "_" + str(alpha) + ".csv", a, delimiter=",")

    return np.mean(scores[:, 0]), np.mean(scores[:, 1]), np.mean(scores[:, 2]), np.mean(scores[:, 3])

def clean(input):
    length = len(input.data)
    # print (input.data)
    data_set = np.array(input.data)
    target_vals = np.array(input.target)
    # using the SGD model
    vectorizer = TfidfVectorizer(
        stop_words='english', ngram_range=(1, 2), min_df=3, max_features=50000)
    return (data_set, target_vals, vectorizer, length)

def NG():
    loss_types = ['hinge','log']
    for loss_type in loss_types:
        dirc = '../../output/20_newsgroups/ath.v.chs/outputs/' + loss_type 
        files = os.listdir(dirc)
        for path in files:
            alpha = float(path.split('.csv')[0])
            print alpha
            run_NG(alpha,'MAX',loss_type)
            print alpha
    
def get_NG(alpha,loss):
    my_data = np.genfromtxt('cv/'+ loss + '_' + str(alpha) + '.csv' , delimiter=',')
    return my_data[0],my_data[1],my_data[2],my_data[3]
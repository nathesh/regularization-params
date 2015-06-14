from data import text_label
import csv
import numpy as np
def get_all():
    # data => title and abstract; target => postive or negative
    data_path = '/home/thejas/Documents/python/regularization-params/data/systematic_review/dst.csv'
    target_path = '/home/thejas/Documents/python/regularization-params/data/systematic_review/DST_l1_relevant.txt'
    data_file = open(data_path, 'rb')
    dst = csv.reader(data_file)
    pos = open(target_path, 'rb').read().split('\n')
    #print 'Here1'
    dt = []
    target = []
    for row in dst:
       # print 'Here!'
        title = row[1]
        abstract = row[3]
        dt.append(title+abstract)
        pmid = row[4]
        if pmid in pos:
            target.append('1')
        else:
            target.append('0')
   
    data = text_label()
    data.add_all(dt, target)
    #print 'shit!', len(dt), len(target)
    return data

import os
from data import text_label 



def run():
    s = '../../../data/newsgroups/20_newsgroups/'
    files = os.listdir(s)
    files.sort()
    fi = files.pop(0)
    s_c = s + fi + '/'
    paths = os.listdir(s_c)
    dt = text_label()
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
    print len(dt.data)
    return dt

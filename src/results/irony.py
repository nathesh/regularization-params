import sqlite3
from random import randint
from data import text_label 

def get_labeled_thrice_comments(cursor):
    cursor.execute(
        '''select comment_id from irony_label group by comment_id having count(distinct labeler_id) >= 3;'''
    )
    thricely_labeled_comment_ids = _grab_single_element(cursor.fetchall())
    print "%s comments have been labeled by >= 3 people" % len(thricely_labeled_comment_ids)
    return thricely_labeled_comment_ids


def get_all_comments_from_subreddit(subreddit, cursor):
    # all_comment_ids = get_labeled_thrice_comments()
    # subreddits = _get_subreddits(all_comment_ids)
    # filtered =
    cursor.execute(
        '''select distinct comment_id from irony_label where
                comment_id in (select id from irony_comment where subreddit='%s');''' %
        subreddit
    )
    subreddit_comments = _grab_single_element(cursor.fetchall())
    return list(set(subreddit_comments))


def _grab_single_element(result_set, COL=0):
    return [x[COL] for x in result_set]

def get_conservatives_liberal(type_vote):
	'''
	return data
	'''
	#print 'Here in get_conservatives_liberal'
	db_path = '../../	data/irony/ironate.db'
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()
	labeled_comment_ids = get_labeled_thrice_comments(cursor)
	# print labeled_comment_ids
	conservative_comment_ids = list(set([c_id for c_id in
	                                     get_all_comments_from_subreddit("Conservative",cursor) if c_id in labeled_comment_ids]))
	liberal_comment_ids = list(set([c_id for c_id in
	                                get_all_comments_from_subreddit("progressive", cursor) if c_id in labeled_comment_ids]))


	if type_vote == 'MAX':
		cursor.execute('SELECT segment_id,text,MAX(label) FROM irony_commentsegment,irony_label \
		    		WHERE irony_label.segment_id = irony_commentsegment.id AND irony_label.forced_decision = 0 \
					GROUP BY irony_commentsegment.id having count(label)>2')
	else:
		cursor.execute('SELECT segment_id,text,SUM(label) FROM irony_commentsegment,irony_label \
		    		WHERE irony_label.segment_id = irony_commentsegment.id AND irony_label.forced_decision = 0 \
					GROUP BY irony_commentsegment.id having count(label)>2')
	fetched_all = cursor.fetchall() 
	liberal = [fetched_all[ids] for ids in liberal_comment_ids]
	conservative = [fetched_all[ids] for ids in conservative_comment_ids]
	liberal_conservative = liberal + conservative
	#print liberal_conservative[0]
	texts = [text for _,text,_ in liberal_conservative]
	labels = [label for _,_,label in liberal_conservative]
	labels = [1  if label > 0 else -1 for label in labels]
	#print labels[0]
	data = text_label()
	data.add_all(texts,labels)
	#print 'Here and'
	return data

def get_all(type_vote):
    db_path = '../../../data/irony/ironate.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if type_vote == 'MAX':
	    cursor.execute('SELECT text,MAX(label) FROM irony_commentsegment,irony_label \
	    WHERE irony_label.segment_id = irony_commentsegment.id AND irony_label.forced_decision = 0 \
	    GROUP BY irony_commentsegment.id having count(label)>2')

    text_labels = cursor.fetchall()
    text = map(lambda x: x[0], text_labels)
    labels = map(lambda x: 1 if x[1] > 0 else 0, text_labels)
    data = text_label()
    data.add_all(text, labels)
    return data


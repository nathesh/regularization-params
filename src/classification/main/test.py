import sqlite3 
from random import randint
db_path = '../../../data/irony/ironate.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('SELECT text,MAX(label) FROM irony_commentsegment,irony_label \
WHERE irony_label.segment_id = irony_commentsegment.id AND irony_label.forced_decision = 0 \
GROUP BY irony_commentsegment.id having count(label)>2')
text_labels = cursor.fetchall()
text = map(lambda x: x[0], text_labels)
labels = map(lambda x: 1 if x[1] > 0 else 0, text_labels)
sum1 = 0
for x in labels:
	if x==1:
		sum1 += 1

print sum1, len(labels)
#print text_labels
print 'trail 2'
cursor.execute('SELECT text,SUM(label) FROM irony_commentsegment,irony_label \
WHERE irony_label.segment_id = irony_commentsegment.id AND irony_label.forced_decision = 0 \
GROUP BY irony_commentsegment.id having count(label)>2')
text_labels = cursor.fetchall()
text = map(lambda x: x[0], text_labels)
labels = map(lambda x: 1 if x[1] > 0 else 0, text_labels)
sum1 = 0
for x in labels:
	if x==1:
		sum1 += 1

print sum1,len(labels)

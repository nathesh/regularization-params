import sqlite3 
from random import randint
db_path = '../../../data/irony/ironate.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('SELECT text,label FROM irony_commentsegment,irony_label \
WHERE irony_label.segment_id = irony_commentsegment.id')
text_labels = cursor.fetchall()
x =randint(0,32635-1)
print x
for tl in text_labels:
	if(tl[0] == text_labels[x][0]):
		print tl[1]

print 'TRIAL 1 DONE!'

cursor.execute('SELECT text,SUM(label) FROM irony_commentsegment,irony_label \
WHERE irony_label.segment_id = irony_commentsegment.id GROUP BY irony_commentsegment.id')
text_labels = cursor.fetchall()
print len(text_labels)
print text_labels[x][1]

print 'TRIAL 2 DONE!'
cursor.execute('SELECT text,SUM(label),forced_decision FROM irony_commentsegment,irony_label \
WHERE irony_label.segment_id = irony_commentsegment.id AND irony_label.forced_decision = 0 \
GROUP BY irony_commentsegment.id')
text_labels = cursor.fetchall()
print len(text_labels)
print text_labels[0][1]
print text_labels[0][2], "forced label"
for t in text_labels:
	if t[2] == 1:
		print 'FUCK'

print 'Trail 4'

cursor.execute('SELECT id,forced_decision FROM irony_label')
text_labels = cursor.fetchall()
sum_t = 0;
print len(text_labels)
for t in text_labels:
	#print 'id', t[0]
	if t[1] == 1:
		sum_t += 1

print len(text_labels)-sum_t
#print text_labels

import os 
import re 

path = '../../data/20_newsgroups/'
files = os.listdir(path)
files.sort()
first_file = files.pop(0) # all the categories 
path_s = path + first_file + "/"
#files = os.listdir(path_s)
for f in files:
	path_s = path + f+'/'
	paths = os.listdir(path_s)
	print paths
	#continue 
	for fil in paths:
		path_f = path_s + fil 
		file_open = open(path_f, 'r') 
		data = file_open.read()
		data = (data.decode('latin1'))
		text = data

		# Part 1 
		_before, _blankline, after = text.partition('\n\n')

		text = after 

		# part 2 
		_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
		                  r'|^In article|^Quoted from|^\||^>)')
		good_lines = [line for line in text.split('\n')
		              if not _QUOTE_RE.search(line)]
		text = '\n'.join(good_lines)

		# part 3 
		lines = text.strip().split('\n')
		for line_num in range(len(lines) - 1, -1, -1):
		    line = lines[line_num]
		    if line.strip().strip('-') == '':
		        break

		if line_num > 0:
		    print '\n'.join(lines[:line_num])
		else:
		    print text

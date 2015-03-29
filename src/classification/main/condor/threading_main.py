#!/usr/local/bin/python
import numpy as np 
import threading 
from subprocess import call

def worker(num):
	call(["python","main_T.py",str(num)])


alpha_vals = np.linspace(.000001, .01, 20)  # input
threads = []
f = open('run.py','w')
for alpha in alpha_vals:
	t = threading.Thread(target=worker, args=(alpha,))
	threads.append(t)
	t.start()
	#f.write("call([python,main.py,"+ str(alpha) + "]"+ "\n")


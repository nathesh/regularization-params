import numpy as np 
alpha_vals = np.linspace(.000001, .01, 20)  # input
f = open('run.py','w')
for alpha in alpha_vals:
	f.write("call([python, main_T.py, " + str(alpha) + '])\n')


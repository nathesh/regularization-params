import subprocess as s 
import numpy as np 
a = np.linspace(.001, .00001, 60)  # input
for alpha in a:
   # print type(str(alpha)),alpha
    s.call(["python","test.py",str(alpha)])

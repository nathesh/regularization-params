from multiprocessing import Process, Value, Array
import numpy as np
def f(n, a):
    #n.value = 3.12
    a.value = a.value*-1
    print a.value

if __name__ == '__main__':
    num = Value('d', 0.0)
    nums = np.linspace(.000001, .01, 20)
    for x in nums:
        num_2 = Value('d', x)
        p = Process(target=f, args=(nums, num_2))
        p.start()
        p.join()

        print num.value
        #print num_2.value
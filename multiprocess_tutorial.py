#! /usr/bin/env python 3

import numpy as np
from multiprocessing import Process
# import time
from time import time, sleep, perf_counter

def this_thread(start,stop,step):
    for index in range(start,stop,step):
        if index % 10000000 ==0:
            print(index)


if __name__ == '__main__':

    start_time = perf_counter()

    # start parallel process
    p = Process(target=this_thread, args=(100000001,-1,-1))
    p.start()
    # p.join()


    this_thread(0,100000001,1)

    final_time = perf_counter()
    print(final_time-start_time)
#! /usr/bin/env python 3

import numpy as np
from multiprocessing import Process, Array, Value, Queue
# import time
from time import time, sleep, perf_counter

def thread(start,stop,step):
    for index in range(start,stop,step):
        if index % 10000000 == 0:
            # print(index)
            pass

def share_process(start,stop,step,queue):
    '''Illustrate exchanging information between processes'''

    for index in range(start,stop,step):
        if index % 10000000 == 0:
            # queue.put(index)
            pass
            # print(index)

def shared_memory(start,stop,step,shared_value):
    '''Illustrate memory sharing'''

    for index in range(start,stop,step):
        if index % 10000000 == 0:
            shared_value.value=index


if __name__ == '__main__':

    ##############################################################################
    # Simple parallel process

    # start parallel process
    p = Process(target=thread, args=(100000001,-1,-1))
    start_time = perf_counter()

    p.start()
    # p.join() # blocks the main thread until the parallel process if finished

    # thread(0,100000001,1)
    for index in range(0,100000001,1):
        if index % 10000000 == 0:
            # parallel_value = q.get()
            pass
            # print(f"Wow! The values are {parallel_value} and {index}!")

    final_time = perf_counter()
    print(final_time-start_time)

    # ##############################################################################
    # # Parallel processing with communication channel
    # q = Queue()

    # p = Process(target=share_process, args=(100000001,-1,-1,q))

    # start_time = perf_counter()

    # p.start()

    # for index in range(0,100000001,1):
    #     if index % 10000000 == 0:
    #         # parallel_value = q.get()
    #         pass
    #         # print(f"Wow! The values are {parallel_value} and {index}!")

    # final_time = perf_counter()
    # print(final_time-start_time)

    # ###############################################################################
    # # Parallel processing with shared memory

    # # define value to be shared across the processes
    # number = Value('d',0.0)

    # # instantiate parallel process
    # p = Process(target=shared_memory,args=(100000001,-1,-1,number))

    # start_time = perf_counter()

    # p.start()

    # for index in range(0,100000001,1):
    #     if index % 10000000 == 0:
    #         parallel_value = number.value
    #         # print(f"Wow! The values are {parallel_value} and {index}!")

    # final_time = perf_counter()
    # print(final_time-start_time)
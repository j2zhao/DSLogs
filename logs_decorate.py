
import functools
import utils
import random
import string
import shutil
import os
import inspect
import pickle
from logged_array import LoggedNDArray
import numpy as np
import time

file_name = './logs/log.txt'
directory = './logs'

def write_log(file, timestamp, function_name, frame_index = -1, input_ids = '', output_ids = ''): 
    frame = inspect.stack()[frame_index][0]
    info = inspect.getframeinfo(frame)
    context = ','.join([str(info.function), str(info.lineno)])
        
    input_ids = str(input_ids)
    output_ids = str(output_ids)
    log = {'time': timestamp, 'context': context, 'function_name': function_name, 'input_ids': input_ids, 'output_ids': output_ids}
    log = str(log)
    log = log + '\n'
    file.write(log)


def write_child_log(file, time, parent_ids, child_ids):
    if isinstance(parent_ids, list):
        parent_ids = ','.join(parent_ids)
    if isinstance(child_ids, list):
        parent_ids = ','.join(child_ids)
    log = '{};relation;{};{}\n'.format(time, parent_ids, child_ids)
    file.write(log)

def write_new_log(file, time, id):
    log = '{};new;{}\n'.format(time, id)
    file.write(log)

def log_attribute(func):
    @functools.wraps(func)
    def function(ref):
        output_ids = []
        output = func(ref)
        if isinstance(output, LoggedNDArray):
            output_ids.append(output.get_id())
        write_log(ref.file, str(time.time()), func.__name__, input_ids = ref.get_id(), output_ids=output_ids)
        return output
    return function

def _log_args(arg):
    if isinstance(arg, LoggedNDArray):
        return arg.get_id()
    elif isinstance(arg, np.ndarray):
        id_file = str(id(arg)) + '_' + utils.rand_string(10)
        id_ = str((arg.shape, id_file))
        array_path = os.path.join(directory, id_file + '.npy')
        with open(array_path, 'wb') as file:
            np.save(file, arg)
        return id_
    else: 
        return (id(arg), arg)

def _log_kwargs(arg):
    key, val = arg
    if isinstance(val, LoggedNDArray):
        return (key, val.get_id())
    
    elif isinstance(val, np.ndarray):
        id_file = str(id(val)) + '_' + utils.rand_string(10)
        id_ = str((val.shape, id_file))
        array_path = os.path.join(directory, id_file + '.npy')
        with open(array_path, 'wb') as file:
            np.save(file, val)
        return (key, id_)
    else:   
        return (key, (id(val), val))

def log_function(func):
    @functools.wraps(func)
    def function(ref, *args, **kwargs):
        input_ids = [str(ref.get_id())]
        for arg in args:
            input_ids.append(utils.coll_function(_log_args, arg))
        # if func.__name__ == '__array_function__':
        #     print(input_ids)
        for arg in kwargs:
            input_ids.append(utils.coll_function(_log_kwargs, [arg, kwargs[arg]]))
        
        output = func(ref, *args, **kwargs)
        # do all outputs
        
        output_ids = utils.coll_function(_log_args, output)
        if isinstance(output, tuple):
            output_ids = list(output_ids)    
        else:
            output_ids = [output_ids]       

        write_log(ref.file, str(time.time()), func.__name__, input_ids = input_ids, output_ids=output_ids)
        return output
    return function

import numpy as np
import typing
import random
import string
import shutil
import os
import inspect
import pickle
import time
import functools
import utils

from numpy.core.arrayprint import str_format
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from numpy.lib.arraysetops import isin

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


class LoggedNDArray(np.ndarray):

    def get_id(self, index = None):
        if index != None:
            id_ = str(id(self)) + '_' + index
        else:
            id_ = id(self)
        id_ = (self.shape, id_)
        return id_

    @property
    def __array_priority__(self) -> typing.Any:
        return 100

    @property
    @log_attribute
    def T(self) -> typing.Any:
        return super().T

    @log_function
    def fill(self,  *args, **kwargs) -> typing.Any:
        return super().fill( *args, **kwargs)
    
    @log_function
    def item(self, *args, **kwargs) -> typing.Any:
        return super().item( *args, **kwargs)
    
    @log_function
    def itemset(self, *args, **kwargs) -> typing.Any:
        return super().itemset( *args, **kwargs)

    @log_function
    def reshape(self, *args, **kwargs) -> typing.Any:
        return super().reshape(*args, **kwargs)
    
    @log_function
    def resize(self, *args, **kwargs) -> typing.Any:
        return super().resize( *args, **kwargs)

    @log_function
    def transpose(self, *args, **kwargs) -> typing.Any:
        return super().transpose( *args, **kwargs)

    @log_function
    def swapaxes(self, *args, **kwargs) -> typing.Any:
        return super().swapaxes(*args, **kwargs)
    
    @log_function
    def flatten(self,  *args, **kwargs) -> typing.Any:
        return super().flatten( *args, **kwargs)

    @log_function
    def ravel(self,  *args, **kwargs) -> typing.Any:
        return super().ravel( *args, **kwargs)

    @log_function
    def squeeze(self,  *args, **kwargs) -> typing.Any:
        return super().squeeze( *args, **kwargs)

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        #if obj is None:
        self.file = open(file_name, 'a+')
        if isinstance(obj, LoggedNDArray):
            id_ = str(obj.get_id())
            self.write_log = getattr(obj, 'write_log', True)
            if self.write_log:
                write_child_log(self.file, time.time(), id_, str(self.get_id()))
        else:
            self.write_log = True
            write_new_log(self.file, time.time(), str(self.get_id()))
    
    def __array__(self, dtype=...) -> typing.Any:
        if dtype == self.dtype:
            return self
        arr = self.view(np.ndarray)
        result = LoggedNDArray(arr.__array(dtype))
        write_child_log(self.file, time.time(), str(self.get_id()), str(self.result.get_id()))
        return result

    @log_function
    def take(self,  *args, **kwargs) -> typing.Any:
        return super().take( *args, **kwargs)
        
        
    @log_function
    def put(self, *args, **kwargs) -> typing.Any:
        return super().put( *args, **kwargs)

    @log_function
    def repeat(self,  *args, **kwargs) -> typing.Any:
        return super().repeat( *args, **kwargs)

    @log_function
    def choose(self, *args, **kwargs) -> typing.Any:
        return super().choose( *args, **kwargs)

    @log_function
    def sort(self, *args, **kwargs) -> typing.Any:
        return super().sort(*args, **kwargs)

    @log_function
    def argsort(self,  *args, **kwargs) -> typing.Any:
        return super().argsort( *args, **kwargs)

    @log_function
    def partition(self,  *args, **kwargs) -> typing.Any:
        return super().partition( *args, **kwargs)
    
    @log_function
    def argpartition(self,  *args, **kwargs) -> typing.Any:
        return super().argpartition(*args, **kwargs)

    @log_function
    def searchsorted(self,  *args, **kwargs) -> typing.Any:
        return super().searchsorted(*args, **kwargs)

    @log_function
    def nonzero(self) -> typing.Any:
        return super().nonzero()

    @log_function
    def compress(self, *args, **kwargs) -> typing.Any:
        return super().compress( *args, **kwargs)

    @log_function
    def diagonal(self, *args, **kwargs) -> typing.Any:
        return super().diagonal( *args, **kwargs)
    
    @log_function
    def max(self,  *args, **kwargs) -> typing.Any:
        return super().max( *args, **kwargs)

    @log_function
    def argmax(self,  *args, **kwargs) -> typing.Any:
        return super().argmax( *args, **kwargs)

    @log_function
    def min(self, *args, **kwargs) -> typing.Any:
        return super().min( *args, **kwargs)

    @log_function
    def argmin(self,  *args, **kwargs) -> typing.Any:
        return super().argmin( *args, **kwargs)

    @log_function
    def ptp(axis=None,  *args, **kwargs) -> typing.Any:
        return super().ptp( *args, **kwargs)

    @log_function
    def clip(self, *args, **kwargs) -> typing.Any:
        return super().clip( *args, **kwargs)
    
    @log_function
    def trace(self,  *args, **kwargs) -> typing.Any:
        return super().trace( *args, **kwargs)

    @log_function
    def sum(self,  *args, **kwargs)-> typing.Any:
        return super().sum( *args, **kwargs)

    @log_function
    def cumsum(self,  *args, **kwargs) -> typing.Any:
        return super().cumsum( *args, **kwargs)

    @log_function
    def mean(self,  *args, **kwargs) -> typing.Any:
        return super().mean( *args, **kwargs)

    @log_function
    def var(self,  *args, **kwargs) -> typing.Any:
        return super().var( *args, **kwargs)
    
    @log_function
    def std(self,  *args, **kwargs) -> typing.Any:
        return super().std( *args, **kwargs)

    @log_function
    def prod(self,  *args, **kwargs) -> typing.Any:
        return super().prod( *args, **kwargs)
    
    @log_function
    def cumprod(self,  *args, **kwargs) -> typing.Any:
        return super().cumprod( *args, **kwargs)

    @log_function
    def __getitem__(self,  *args, **kwargs) -> typing.Any:
        return super().__getitem__( *args, **kwargs)

    @log_function
    def __setitem__(self,  *args, **kwargs) -> None:
        return super().__setitem__( *args, **kwargs)

    # do stuff
    # @log_function
    # def __array_function__(self, func, types, *args, **kwargs) -> typing.Any:
    #     args_ = []
    #     kwargs_ = {}
    #     types_ = []
    #     for ty in types:
    #         if not isinstance(ty, type):
    #             raise NotImplemented('types variable collection type not supported')
    #         if ty == LoggedNDArray:
    #             types_.append(np.ndarray)
    #         else:
    #             types_.append(ty)
        
    #     for input in args:
    #         if isinstance(input, LoggedNDArray):
    #             args_.append(input.view(np.ndarray))
    #         else:
    #             args_.append(input)
        
    #     for key, val in kwargs.items():
    #         if isinstance(val, LoggedNDArray):
    #             kwargs_[key] = val.view(np.ndarray)
    #         else:
    #             kwargs_[key] = val

    #     output =  super().__array_function__(func, types_, *args_, **kwargs_)
        
    #     if isinstance(output, np.ndarray):
    #         output = output.view(LoggedNDArray)
        
    #     elif isinstance(output, tuple):
    #         output_ = []
    #         for out in output:
    #             if isinstance(out, np.ndarray):
    #                 output_.append(out.view(LoggedNDArray))
    #             else:
    #                 output_.append(out)
    #         output = output_

    #     return output

    # do stuff
    @log_function
    def __array_function__(self, func, types, *args, **kwargs) -> typing.Any:
        return super().__array_function__(func, types, *args, **kwargs)
    
    @log_function
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        kwargs_ = {}
        for input in inputs:
            if isinstance(input, LoggedNDArray):
                args.append(input.view(np.ndarray))
            else:
                args.append(input)

        for key, val in kwargs.items():
            if isinstance(val, LoggedNDArray):
                kwargs_[key] = val.view(np.ndarray)
            else:
                kwargs_[key] = val

        output =  super().__array_ufunc__(ufunc, method,
                                                 *args, **kwargs_)
        
        if isinstance(output, np.ndarray):
            output = output.view(LoggedNDArray)
        
        elif isinstance(output, tuple):
            output_ = []
            for out in output:
                if isinstance(out, np.ndarray):
                    output_.append(out.view(LoggedNDArray))
                else:
                    output_.append(out)
            output = output_

        return output





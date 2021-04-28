import numpy as np
from logged_array_2 import LoggedNDArray
import shutil
import os
import copy

shutil.rmtree('logs')
os.makedirs('logs')

arr = LoggedNDArray(np.arange(100))
arr_2 = LoggedNDArray(np.arange(100))
#np.round(arr)
# arr = LoggedNDArray(np.zeros((10, 10)))
# arr_2 = LoggedNDArray(np.zeros((10,)))
#arr.getfield(np.float64)
#print(arr)
#arr.T
# np.concatenate((arr, arr_2))
# np.tile(arr, 2)
it = np.nditer(arr)
for a in it:
    print(a)
#print(type(arr.T))
# arr.var()
#np.amin(arr) # works#

#np.amax(arr) # works

# np.nanmax(arr) #works


#np.ptp(arr) # doesn't work

#np.percentile(arr, 10) # doesn't work -> consider the implementation -> this is incedibly complicated

#np.nanpercentile(arr, 10) #doesn't work

# np.quantile(arr, 0.4) #doesn't work
#result = np.partition(arr, 3)

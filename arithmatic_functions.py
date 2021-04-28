"""
sample of arithmatic functions -> not direct ufuncs
-> it seems like 90% of this will be covered by ufuncs
"""

import numpy as np
from logged_array import LoggedNDArray


# # trig functions -> unwrap -> hmm not
# y = LoggedNDArray(np.ones((10, 10, 10))*(np.pi/2))

# a = LoggedNDArray(np.ones((10, 10, 10))*4)
# b = LoggedNDArray(np.ones((10, 10, 10))*3)
# z = np.sin(y)
# w = np.cos(y)
# np.arctan2(a, b)
# np.hypot(a, b)


# # rounding functions -> interesting question: how do we deal with this?

# x = LoggedNDArray(np.ones((10, 10, 10))*np.pi)
# np.around(x) #equivalent to round_
# np.rint(x)
# np.fix(x)
# np.floor(x)
# np.ceil(x)
# np.trunc(x)

# # # sum, prod, etc. between elements? 
# a = LoggedNDArray(np.ones((10, 10)))
# # b = LoggedNDArray(np.asfarray([]))

# d = np.prod(a)

# c = np.prod(a)

arr_1 = LoggedNDArray(np.ones((10,)))
arr_2 = LoggedNDArray(np.ones((10,)))
#arr = np.ones((10, 10))

#np.amin(arr) #minimum functions -> don't know which one

#np.nanstd(arr) # really look at this

np.correlate(arr_1, arr_2)
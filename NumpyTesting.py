#! /usr/env python
# -*- coding: utf8 -*-

import numpy as np

x = np.array([[1,2,3,4],[5,6,7,8],[11,12,13,14],[15,16,17,18]], np.float)
y= np.ndarray(shape=(5,5), dtype='float16')

print x

x.itemset((0,2),0.88763)
print x

l=234
l2 = [1,2,3,4]
print isinstance(l,int)
print isinstance(l2, list)
print type(l)=='type int'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
 
class data_record(object):
    # This class define a NxN matrix to store the classifier result.
    # And the row for matrix means predicted result and the col means label.
    def __init__(self, path, head=[]):        
        # Predict of Truth
        self.head = ['epoch']
        self.data = []
        for h in head:
            self.head.append(h)
            
        self.path = path
        with open(self.path, 'w') as f:
            f.write(','.join(np.array(self.head, dtype=np.str).tolist())+'\n')
        
    def add_record(self, data):
        self.data.append(data)        
        with open(self.path, 'a+') as f:
            f.write(','.join(np.array(data, dtype=np.str).tolist())+'\n')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

def aug_from_list(flist, times=1):
    results=[];
    for i in range(times):
        results += flist
    return results
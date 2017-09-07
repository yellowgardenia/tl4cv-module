from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import xml.dom.minidom
from sklearn.metrics import confusion_matrix
 
class classifier(object):
    # This class define a NxN matrix to store the classifier result.
    # And the row for matrix means predicted result and the col means label.
    def __init__(self, num_classes):        
        # Predict of Truth
        self.PofT = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        
    def clear(self):
        self.PofT = np.zeros((self.num_classes, self.num_classes))
        
    def load_result(self, input_mat):
        if input_mat.shape[0] == input_mat.shape[1]:
            self.PofT = input_mat
            self.num_classes = input_mat.shape[0]
            return True
        else:
            return False
        
    def load_from_list(self, pred, label):
        self.PofT = confusion_matrix(pred, label)
        self.PofT = np.array(self.PofT, dtype=np.float)
            
    def add_result(self, pred, label):
        # pred and label need to begin from 0
        try:
            self.PofT[pred, label] += 1
        except:
            print('Error: Input need to be integer from 0 to num_classes-1.')
            
    def calculate_acc(self):
        # calculate total accuracy
        total_obj = self.PofT.sum()
        dialog = np.multiply(self.PofT, np.eye(self.num_classes))
        total_true = dialog.sum()
        
        if total_obj == 0:
            return 0
        else:
            acc = 1.0*total_true/total_obj
            return acc
    
    def calculate_acc_n(self):
        # calculate accuracy for each class
        total_obj = self.PofT.sum(axis=0)
        dialog = np.multiply(self.PofT, np.eye(self.num_classes))
        total_true = dialog.sum(axis=0)
        
        idx = [idx for (idx, val) in enumerate(total_obj) if val == 0]
        total_obj[idx] += 1e-9
        acc = total_true/total_obj
        return acc
    
    def calculate_acc_nxn(self):
        # calculate accuracy for each pred-label
        total_obj = self.PofT.sum(axis=0)
        idx = [idx for (idx, val) in enumerate(total_obj) if val == 0]
        total_obj[idx] += 1e-9
        acc = self.PofT/total_obj
        return acc
    
    def array2str(self, arr):
        txt = ''
        
        if arr.ndim == 1:
            #m = arr.shape
            txt = ','.join(np.str(i) for i in arr)
        elif arr.ndim == 2:
            txt = '\n'.join(','.join(np.str(col) for col in row) for row in arr)
            #m, _ = arr.shape
            #for row in range(m):
            #    s = ','.join(str(i) for i in arr[row])
            #    txt += (s+'\n')
        return txt
    
    def write_xml(self, xml_path):
        # create an empty file
        doc = xml.dom.minidom.Document()
        # create root
        root = doc.createElement('Results')
        # add root attribute
        now = datetime.datetime.now()
        root.setAttribute('Date', now.strftime('%Y-%m-%d %H:%M:%S'))
        root.setAttribute('Object', 'ClassifierStatistic')
        root.setAttribute('n_classes', str(self.num_classes))
        root.setAttribute('n_files', str(np.int(self.PofT.sum())))
        # add root to file
        doc.appendChild(root)
        
        nodeEachClassNum = doc.createElement('EachClassNum')
        nodeEachClassNum.appendChild(doc.createTextNode(self.array2str(self.PofT.sum(axis=0))))
        root.appendChild(nodeEachClassNum)
        
        nodeResultNxN = doc.createElement('ResultNxN')
        nodeResultNxN.appendChild(doc.createTextNode(self.array2str(self.PofT)))
        root.appendChild(nodeResultNxN)
        
        nodeAccuracy = doc.createElement('Accuracy')
        nodeAccuracy.appendChild(doc.createTextNode(str(self.calculate_acc())))
        root.appendChild(nodeAccuracy)
        
        nodeAccuracyN = doc.createElement('AccuracyN')
        nodeAccuracyN.appendChild(doc.createTextNode(self.array2str(self.calculate_acc_n())))
        root.appendChild(nodeAccuracyN)
        
        nodeAccuracyNxN = doc.createElement('AccuracyNxN')
        nodeAccuracyNxN.appendChild(doc.createTextNode(self.array2str(self.calculate_acc_nxn())))
        root.appendChild(nodeAccuracyNxN)

        # write xml
        fp = open(xml_path, 'w')
        doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
        
    def load_xml(self, xml_path):
        DOMTree = xml.dom.minidom.parse(xml_path)
        collection = DOMTree.documentElement
        
        result_nxn = collection.getElementsByTagName("ResultNxN")[0]
        r_data = result_nxn.childNodes[0].data
        
        self.PofT = np.array([([col for col in row.split(',')]) for row in r_data.split('\n')], dtype=np.float)
        self.num_classes, _ = self.PofT.shape
        
        #row = r_data.split('\n')
        #for i in range(self.num_classes):
        #    col = row[i].split(',')
        #    for j in range(self.num_classes):
        #        self.PofT[i, j] = np.float(col[j])
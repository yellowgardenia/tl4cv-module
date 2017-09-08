from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import xml.dom.minidom
 
class predict_result(object):
    # This class define a MxN matrix to store the prediction results.
    # And the M for matrix means M objects and the N means num_classes.
    def __init__(self, num_classes):
        # result list
        self.rl = []
        # label list
        self.ll = []
        # name list
        self.nl = []
        
        self.num_classes = num_classes
        self.num_obj = 0
        
    def clear(self):
        self.rl = []
        self.ll = []
        self.num_obj = 0
        
    def load_from_list(self, pred, name, label):
        # pred: [[0.1, 0.2, 0.3, 0.2, 0.1], ...]
        # label: [0, ...]
        # name: ['name', ...]
        self.rl = pred
        self.ll = label
        self.nl = name
        self.num_obj = len(pred)
            
    def add_result(self, pred, name, label):
        # pred: [0.1, 0.2, 0.3, 0.2, 0.1]
        # label: 0
        # name: 'name'
        self.rl += [pred]
        self.ll += [label]
        self.nl += [name]
        self.num_obj += 1
        
    def predmax(self):
        return np.argmax(self.rl, axis=1)
        
    def array2str(self, arr):
        txt = ','.join(np.str(i) for i in arr)
        return txt
    
    def write_xml(self, xml_path):
        # create an empty file
        doc = xml.dom.minidom.Document()
        # create root
        root = doc.createElement('Results')
        # add root attribute
        now = datetime.datetime.now()
        root.setAttribute('Date', now.strftime('%Y-%m-%d %H:%M:%S'))
        root.setAttribute('Object', 'Prediction')
        root.setAttribute('n_classes', np.str(self.num_classes))
        root.setAttribute('n_files', np.str(self.num_obj))
        # add root to file
        doc.appendChild(root)
        
        for i in range(self.num_obj):
            p = self.rl[i]
            name = self.nl[i]
            label = self.ll[i]
            
            obj = doc.createElement('obj')
            root.appendChild(obj)
            
            fname = doc.createElement('name')
            fname_text = doc.createTextNode(name)
            fname.appendChild(fname_text)
            obj.appendChild(fname)
            
            flabel = doc.createElement('label')
            flabel_text = doc.createTextNode(np.str(label))
            flabel.appendChild(flabel_text)
            obj.appendChild(flabel)
            
            fpred = doc.createElement('pred')
            fpred_text = doc.createTextNode(self.array2str(p))
            fpred.appendChild(fpred_text)
            obj.appendChild(fpred)

        # write xml
        with open(xml_path, 'w') as fp:
            doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
        
    def load_xml(self, xml_path):
        DOMTree = xml.dom.minidom.parse(xml_path)
        root = DOMTree.documentElement
        
        self.num_classes = np.int(root.getAttribute('n_classes'))
        #self.num_obj = np.int(root.getAttribute('n_files'))
        
        obj_list = root.getElementsByTagName('obj')
        
        for obj in obj_list:
            r_data = obj.getElementsByTagName('pred')[0].childNodes[0].nodeValue
            p = np.array([col for col in r_data.split(',')], dtype=np.float).tolist()
            n = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
            l = np.int(obj.getElementsByTagName('label')[0].childNodes[0].nodeValue)
            
            self.add_result(p, n, l)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import xml.dom.minidom
from predict_result import predict_result
 
class feature_result(predict_result):
    # This class define a MxN matrix to store the prediction results.
    # And the M for matrix means M objects and the N means feature num.
    def __init__(self, num_features):
        predict_result.__init__(self, num_features)
        self.num_features = num_features
    
    def write_xml(self, xml_path):
        # create an empty file
        doc = xml.dom.minidom.Document()
        # create root
        root = doc.createElement('Results')
        # add root attribute
        now = datetime.datetime.now()
        root.setAttribute('Date', now.strftime('%Y-%m-%d %H:%M:%S'))
        root.setAttribute('Object', 'Features')
        root.setAttribute('n_features', np.str(self.num_features))
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
            
            fpred = doc.createElement('feature')
            fpred_text = doc.createTextNode(self.array2str(p))
            fpred.appendChild(fpred_text)
            obj.appendChild(fpred)

        # write xml
        with open(xml_path, 'w') as fp:
            doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
        
    def load_xml(self, xml_path):
        DOMTree = xml.dom.minidom.parse(xml_path)
        root = DOMTree.documentElement
        
        self.num_features = np.int(root.getAttribute('n_features'))
        #self.num_obj = np.int(root.getAttribute('n_files'))
        
        obj_list = root.getElementsByTagName('obj')
        
        for obj in obj_list:
            r_data = obj.getElementsByTagName('feature')[0].childNodes[0].nodeValue
            p = np.array([col for col in r_data.split(',')], dtype=np.float).tolist()
            n = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
            l = np.int(obj.getElementsByTagName('label')[0].childNodes[0].nodeValue)
            
            self.add_result(p, n, l)
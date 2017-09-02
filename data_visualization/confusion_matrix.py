from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')
from classifier_statistics.distribution import classifier

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

class draw_confmat(object):
    # a draw model for classifier
    def __init__(self, num_classes):
        self.cm = classifier(num_classes)
        
    def load_result(self, input_mat):
        self.cm.load_result(input_mat)
        
    def load_from_list(self, pred, label):
        self.cm.load_from_list(pred, label)
        
    def load_xml(self, xml_path):
        self.cm.load_xml(xml_path)
        
    def __plot_confusion_matrix(self, cm, class_name,
                                title='Confusion Matrix',
                                xylabel=['Predicted label', 'True label'],
                                cmap=plt.cm.Blues,
                                transform=True):
        if transform == True:
            cm = np.transpose(cm)
            
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_name))
        plt.xticks(tick_marks, class_name, rotation=45)
        plt.yticks(tick_marks, class_name)
        
        fmt = ('d' if cm.dtype == 'int' else '.2f')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel(xylabel[1] if transform == True else xylabel[0])
        plt.xlabel(xylabel[0] if transform == True else xylabel[1])
        
    def draw(self, cm, class_name, fig_path,
             title='Confusion Matrix',
             xylabel=['Predicted label', 'True label'],
             transform=True):
        # input: class_name = ['c1','c2',...]
        #        fig_path = 'PathtoSave.png'
        np.set_printoptions(precision=2)
        
        if self.cm.num_classes != len(class_name):
            return False
        
        plt.figure()
        self.__plot_confusion_matrix(cm, class_name=class_name, title=title, xylabel=xylabel, transform=transform)
        plt.savefig(fig_path, dpi=300, facecolor='w', edgecolor='w', pad_inches=0.4, bbox_inches='tight')
        plt.show()
        return True
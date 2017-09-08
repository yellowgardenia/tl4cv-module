from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def draw_roc(fig_path, title='ROC Curve',
             xylabel=['False positive rate', 'True positive rate'],
             **kargs):
    for key in kargs:
        val = kargs[key]
        pred = val['pred']
        label = val['label']
        
        fpr_our, tpr_our, _ = roc_curve(label, pred)
        roc_auc_our = auc(fpr_our, tpr_our)
        
        plt.plot(fpr_our, tpr_our, ls='-', lw=2,
                 label='%s (AUC = %0.2f)' % (key, roc_auc_our))
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(xylabel[0], fontsize=20)
    plt.ylabel(xylabel[1], fontsize=20)
    plt.title(title, fontsize=24)
    plt.legend(loc='best', fontsize='large')
    plt.savefig(fig_path, dpi=300, facecolor='w', edgecolor='w', pad_inches=0.4, bbox_inches='tight')
    plt.show()
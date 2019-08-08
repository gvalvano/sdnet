#  Copyright 2019 Gabriele Valvano
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import time
from sklearn.metrics import confusion_matrix


def eval_dice(seg, gt):
    """
    Returns Sørensen–Dice coefficient for binary masks (only 1 and 0).
    Be aware that the mask must only contain 0 and 1, otherwise you can end up with wrong scores.
    :param seg: segmentation mask
    :param gt: ground truth mask
    :return: dice score
    """
    dice = np.sum(seg[gt == 1])*2.0 / (np.sum(seg) + np.sum(gt))
    return dice


def true_false_positives_negatives(true, pred, normalize=False):
    """
    Returns true positive, false positives, false negatives and true positives.
    :param true: true values
    :param pred: predicted values
    :param normalize: if True the function returns the rates, otherwise it returns the sums (int)
    :return: tn, fp, fn, tp
    """
    y_actual = np.array(true).reshape(-1)
    y_predict = np.array(pred).reshape(-1)
    tn, fp, fn, tp = confusion_matrix(y_actual, y_predict, labels=[0, 1]).ravel()
    if normalize:
        ntot = len(y_actual)
        return tn*1.0/ntot, fp*1.0/ntot, fn*1.0/ntot, tp*1.0/ntot
    return tn, fp, fn, tp


def processing_time(test_function, *args):
    """
    Evaluates the time needed to execute the given function
    :param test_function: function to execute
    :return: processing time and outputs of the function
    """
    start_time = time.time()
    outputs = test_function(*args)
    delta_t = time.time() - start_time
    return delta_t, outputs

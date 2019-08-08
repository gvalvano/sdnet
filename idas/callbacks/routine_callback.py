"""
This is a callback which is always run during the training. 
"""
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

from idas.callbacks.callbacks import Callback
import idas.logger.json_logger as jlogger
import os


def _check_for_sparse_training(cnn, history_logs):
    """ check if the flag for dsd training (eventually added by a DSDCallback()) exists and is True. 
    Retruns has_been_run=True if the try succeeds, False otherwise. """
    has_been_run = False
    try:
        node = jlogger.read_one_node('SPARSE_TRAINING', fname=history_logs)
        if node['done_before']:
            sparsity = node['sparsity']
            beta = node['beta']

            if not cnn.perform_sparse_training:
                # otherwise the learning rate has already been reduced by dsd_callback
                print(" | This network was already trained with Sparsity constraint of \033[94m{0}%\033[0m and a reduced "
                      "learning rate by a multiplying factor beta \033[94m{1}%\033[0m.".format(sparsity * 100, beta * 100))
                print(" | The current learning rate ({0}) will be consequently reduced by the same factor: \033[94m"
                      "corrected learning rate = {1}\033[0m".format(cnn.lr, beta * cnn.lr))
                print(' | - - ')
                cnn.lr = beta * cnn.lr

            has_been_run = True

    except (FileNotFoundError, KeyError):
        pass
    return has_been_run


def _check_for_annealed_lr(cnn, sess, history_logs):
    """ check if the flag for dsd training (eventually added by a DSDCallback()) exists and is True. 
    Retruns has_been_run=True if the try succeeds, False otherwise. """
    has_been_run = False
    try:
        node = jlogger.read_one_node('LR_ANNEALING', fname=history_logs)
        if node['done_before']:

            strategy = node['strategy']
            last_lr = node['last_learning_rate']

            print(" | This network was already trained with a strategy of \033[94m{0}\033[0m and the "
                  "last learning rate was \033[94m{1}\033[0m".format(strategy, last_lr))
            print(" | The learning rate will be consequently set to \033[94m{0}\033[0m".format(last_lr))
            print(' | - - ')
            #cnn.lr = last_lr
            sess.run(cnn.lr.assign(last_lr))

            has_been_run = True

    except (FileNotFoundError, KeyError):
        pass
    return has_been_run


class RoutineCallback(Callback):
    def __init__(self):
        super().__init__()
        self.history_log_file = None

    def on_train_begin(self, training_state, **kwargs):

        print("\nRunning RoutineCallback...")
        if kwargs['cnn'] is None:
            raise Exception

        self.history_log_file = kwargs['history_log_dir'] + os.sep + 'train_history.json'

        # Here perform any check operation on log files, etc.:
        runs = list()
        runs.append(_check_for_sparse_training(kwargs['cnn'], self.history_log_file))
        runs.append(_check_for_annealed_lr(kwargs['cnn'], kwargs['sess'], self.history_log_file))

        # check if the callback has performed any operation:
        has_been_run = runs.count(True) > 0
        if has_been_run:
            print(" >> More then zero actions performed.")
        print("Done.")

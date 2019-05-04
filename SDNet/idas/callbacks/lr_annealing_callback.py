"""
In training deep networks, it is usually helpful to anneal the learning rate over time. Good intuition to have in mind 
is that with a high learning rate, the system contains too much kinetic energy and the parameter vector bounces around 
chaotically, unable to settle down into deeper, but narrower parts of the loss function. Knowing when to decay the 
learning rate can be tricky: Decay it slowly and you’ll be wasting computation bouncing around chaotically with little 
improvement for a long time. But decay it too aggressively and the system will cool too quickly, unable to reach the best 
position it can. There are three common types of implementing the learning rate decay:
  - step decay
  - exponential decay
  - 1/t decay
In practice, the step decay is slightly preferable because the hyperparameters it involves are more interpretable. 
Lastly, if you can afford the computational budget, err on the side of slower decay and train for a longer time.
"""

import numpy as np
import ast
import os
from idas.callbacks.callbacks import Callback
import idas.logger.json_logger as jlogger
import tensorflow as tf


def apply_step_decay(params, t):
    """
    Reduces the learning rate by some factor every few epochs. 
    :param params: parameters for the annealing
    :param t: iteration number (or you can use number of epochs)
    :return: updated learning rate
    """
    lr = params['curr_lr']  # current learning rate
    k = params['k']  # decay factor
    period = params['period']  # period used to anneal
    if (t % period) == 0 and (t != 0):
        return lr * 1. / k
    return lr


def apply_exp_decay(params, t):
    """
    Implements the mathematical form: a = a0 * exp(-k * t).
    :param params: parameters for the annealing
    :param t: iteration number (or you can use number of epochs)
    :return: updated learning rate
    """
    a0 = params['lr0']  # initial learning rate
    k = params['k']  # decay factor
    return a0 * np.exp(-k*t)


def apply_1overt_decay(params, t):
    """
    Implements the mathematical form: a = a0 / (1 + k*t). 
    :param params: parameters for the annealing
    :param t: iteration number (or you can use number of epochs)
    :return: updated learning rate
    """
    a0 = params['lr0']  # initial learning rate
    k = params['k']  # decay factor
    return a0 * 1. / (1 + k*t)


class LrAnnealingCallback(Callback):
    def __init__(self):
        super().__init__()
        # Define variables here because the callback __init__() is called before the initialization of all variables
        # in the graph.
        self.history_log_file = None

    def on_train_begin(self, training_state, **kwargs):
        print("\nAnnealing the learning rate with strategy \033[94m{0}\033[0m".format(kwargs['annealing_strategy']))

        cnn = kwargs['cnn']
        if cnn is None:
            raise Exception

        self.history_log_file = kwargs['history_log_dir'] + os.sep + 'train_history.json'

        try:
            _ = jlogger.read_one_node('LR_ANNEALING', fname=self.history_log_file)
        except (FileNotFoundError, KeyError):
            vals = {'done_before': True,
                    'strategy': kwargs['annealing_strategy'],
                    'parameters': kwargs['annealing_parameters'],
                    'annealing_epoch_delay': kwargs['annealing_epoch_delay'],
                    'last_learning_rate': ast.literal_eval(kwargs['annealing_parameters'])['lr0']}
            jlogger.add_new_node('LR_ANNEALING', vals, fname=self.history_log_file)

        # define update operation:
        up_value = tf.placeholder(tf.float32, None, name='update_lr_value')
        self.update_lr = cnn.lr.assign(up_value, name='update_lr')

    def on_epoch_end(self, training_state, **kwargs):

        cnn = kwargs['cnn']
        if cnn is None:
            raise Exception

        curr_epoch = cnn.g_epoch.eval()

        if curr_epoch > kwargs['annealing_epoch_delay']:

            call_strategy = {'step_decay': apply_step_decay,
                             'exp_decay': apply_exp_decay,
                             '1overT_decay': apply_1overt_decay}

            params = ast.literal_eval(kwargs['annealing_parameters'])  # convert string type to dictionary
            params['curr_lr'] = cnn.lr.eval()  # add current learning rate to the annealing parameters

            # call the right decay method:
            updated_lr = call_strategy[kwargs['annealing_strategy']](params, curr_epoch)

            if updated_lr != params['curr_lr']:
                print("\n\033[94mAnnealing the learning rate with strategy '{0}'... "
                      "New value = {1:0.2e}\033[0m".format(kwargs['annealing_strategy'], updated_lr))

                # cnn.lr = updated_lr
                kwargs['sess'].run(self.update_lr, feed_dict={'update_lr_value:0': updated_lr})

                jlogger.update_node('LR_ANNEALING', sub_key='last_learning_rate', sub_value=updated_lr,
                                    fname=self.history_log_file)

"""
Callback early stopping.
"""
from idas.callbacks.callbacks import Callback


class EarlyStoppingException(Exception):
    """ Raised to early stop the training. """
    pass


class EarlyStoppingCallback(Callback):
    def __init__(self, min_delta=0.01, patience=20):
        """ We want to define a minimum acceptable change (min_delta) in the loss function and a patience parameter
        which once exceeded triggers early stopping. When the loss increases or when it stops decreasing by more than
        min_delta, the patience counter activates. Once the patience counter expires, the callback returns a signal
        (stop = True).
        """
        super().__init__()
        # Define variables here because the callback __init__() is called before the initialization of all variables
        # in the graph.
        assert min_delta > 0
        self.min_delta = min_delta
        self.patience = patience
        self.patience_counter = 0
        self.hist_loss = 1e16

    def on_epoch_end(self, training_state, **kwargs):

        if self.hist_loss - kwargs['es_loss'] > self.min_delta:
            self.patience_counter = 0
            self.hist_loss = kwargs['es_loss']
        else:
            self.patience_counter += 1

            if self.patience_counter > self.patience:
                raise EarlyStoppingException

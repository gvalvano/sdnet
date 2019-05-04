"""
File with the definition of the callbacks.
"""


class Callback(object):
    """ Callback base class. """
    def __init__(self):
        pass

    def on_train_begin(self, training_state, **kwargs):
        pass

    def on_epoch_begin(self, training_state, **kwargs):
        pass

    def on_batch_begin(self, training_state, **kwargs):
        pass

    def on_batch_end(self, training_state, **kwargs):
        pass

    def on_epoch_end(self, training_state, **kwargs):
        pass

    def on_train_end(self, training_state, **kwargs):
        pass


class ChainCallback(Callback):
    """
    Series of callbacks
    """
    def __init__(self, callbacks=None):
        super().__init__()
        if callbacks is None:
            callbacks = []
        self.callbacks = callbacks

    def on_train_begin(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(training_state, **kwargs)

    def on_epoch_begin(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(training_state, **kwargs)

    def on_batch_begin(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(training_state, **kwargs)

    def on_batch_end(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(training_state, **kwargs)

    def on_epoch_end(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(training_state, **kwargs)

    def on_train_end(self, training_state, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(training_state, **kwargs)

    def add(self, callback):
        if not isinstance(callback, Callback):
            raise Exception(str(callback) + " is an invalid Callback object")

        self.callbacks.append(callback)

"""
For the training starting from zero, without fine tuning
"""
import tensorflow as tf
tf.random.set_random_seed(1234)
from model import Model

N_EPOCHS = 10

# ----------------

if __name__ == '__main__':
    print('\n' + '-'*10)
    model = Model()
    model.build()

    model.train(n_epochs=N_EPOCHS)

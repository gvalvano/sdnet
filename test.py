"""
For the test the pre-trained model
"""
from model import Model
import numpy as np

root = 'data/acdc_data/test/'
filename = ''  # chose the file to test

# ----------------

if __name__ == '__main__':
    print('\n' + '-'*10)
    model = Model()
    model.build()

    input_data = np.load(root + filename).astype(np.float32)

    soft_anatomy, hard_anatomy, predicted_mask = model.test(input_data)

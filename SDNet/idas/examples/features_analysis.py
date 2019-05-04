from cnn import ConvNet
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

SAVE_IMAGES = False  # True
DO_TSNE = False  # True


def save_images(matrix, name):
    """ Save correlation and cosine similarity matrices """
    plt.ioff()

    plt.imshow(matrix.astype(np.float32), interpolation='nearest')
    plt.colorbar()
    plt.savefig(name, format='svg', dpi=1000)
    plt.close('all')


if __name__ == '__main__' or True:
    model = ConvNet()
    model.build()

    mnist = input_data.read_data_sets("data/", one_hot=True)
    test_data = mnist.test.images
    test_labels = np.argmax(mnist.test.labels, axis=1)

    inds = test_labels.argsort()
    test_data = test_data[inds]

    img_in = test_data.reshape([-1, 28, 28, 1])

    # latent AE activations
    X = model.eval_layer_activation(img_in, 'layer_name/swish_f32:0')

    Corr = np.corrcoef(X).astype(np.float16)  # np.matmul(X, np.transpose(X))

    np.save('images/Corr.npy', Corr)

    if SAVE_IMAGES:
        save_images(Corr, name='images/corr_matrix.svg')

    # _____________________________ #
    #  T-SNE

    if DO_TSNE:

        features = X
        true_label = test_labels
        X_embedded = TSNE(n_components=2, n_iter=1000).fit_transform(features)
        x_coord, y_coord = X_embedded[:, 0], X_embedded[:, 1]

        fig, (a1) = plt.subplots(1, 1)
        true_label_color = [el for el in true_label]
        a1.scatter(x_coord, y_coord, c=true_label_color, cmap=plt.cm.get_cmap("jet", 10), s=1)
        a1.set_title('true_label clusters')

        plt.show()

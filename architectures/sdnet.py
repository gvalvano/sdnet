import tensorflow as tf
from architectures.unet import UNet
from architectures.modality_encoder import ModalityEncoder
from architectures.decoder import Decoder
from architectures.segmentor import Segmentor
from architectures.layers.rounding_layer import rounding_layer

# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class SDNet(object):

    def __init__(self, n_anatomical_masks, nz_latent, n_classes, is_training, anatomy=None, name='Model'):
        """
        SDNet architecture. For details, refer to:
          "Factorised Representation Learning in Cardiac Image Analysis" (2019), arXiv preprint arXiv:1903.09467
          Chartsias, A., Joyce, T., Papanastasiou, G., Williams, M., Newby, D., Dharmakumar, R., & Tsaftaris, S. A.

        :param n_anatomical_masks: (int) number of anatomical masks (s factors)
        :param nz_latent: (int) number of latent dimensions outputted by modality encoder
        :param n_classes: (int) number of classes (4: background, LV, RV, MC)
        :param is_training: (tf.placeholder, or bool) training state, for batch normalization
        :param anatomy: (tensor) if given, the reconstruction is computed starting from z modality extracted by the
                        input data and the hard anatomy in this argument. Default: compute and use hard anatomy of the
                        input data.
        :param name: variable scope name

        - - - - - - - - - - - - - - - -

        Example of usage:

            # build the sdnet:
            sdnet = SDNet(n_anatomical_masks, nz_latent, n_classes, is_training, name='Model')
            sdnet = sdnet.build(input_data)

            # get soft and hard anatomy:
            soft_a = sdnet.get_soft_anatomy()
            hard_a = sdnet.get_hard_anatomy()

            # get z distribution (output of modality encoder)
            z_mean, z_logvar, sampled_z = sdnet.get_z_distribution()

            # get decoder reconstruction:
            rec = sdnet.get_input_reconstruction()

            # get z estimate given the reconstruction
            z_regress = sdnet.get_z_sample_estimate()

        """
        self.is_training = is_training
        self.name = name

        self.n_anatomical_masks = n_anatomical_masks
        self.nz_latent = nz_latent
        self.n_classes = n_classes
        self.anatomy = anatomy

        self.soft_anatomy = None
        self.hard_anatomy = None
        self.z_mean = None
        self.z_logvar = None
        self.sampled_z = None
        self.reconstruction = None
        self.z_regress = None
        self.pred_mask = None

    def build(self, incoming, reuse=tf.AUTO_REUSE):
        """
        Build the model.
        """
        with tf.variable_scope(self.name, reuse=reuse):
            # - - - - - - -
            # build Anatomy Encoder
            with tf.variable_scope('AnatomyEncoder'):
                unet = UNet(incoming, n_out=self.n_anatomical_masks, is_training=self.is_training, n_filters=64)
                unet_encoder = unet.build_encoder()
                unet_bottleneck = unet.build_bottleneck(unet_encoder)
                unet_decoder = unet.build_decoder(unet_bottleneck)
                unet_output = unet.build_output(unet_decoder)
                self.soft_anatomy = tf.nn.softmax(unet_output)

            with tf.variable_scope('RoundingLayer'):
                self.hard_anatomy = rounding_layer(self.soft_anatomy)

            # - - - - - - -
            # build Modality Encoder
            with tf.variable_scope('ModalityEncoder'):
                anatomy = self.hard_anatomy if (self.anatomy is None) else self.anatomy
                mod_encoder = ModalityEncoder(incoming, anatomy, self.nz_latent, self.is_training).build()
                self.z_mean, self.z_logvar = mod_encoder.get_z_stats()
                self.sampled_z = mod_encoder.get_z_sample()

            # - - - - - - -
            # build Decoder to reconstruct the input given sampled z and the hard anatomy
            with tf.variable_scope('Decoder'):
                decoder = Decoder(self.sampled_z, self.hard_anatomy, self.n_anatomical_masks, is_training=self.is_training).build()
                self.reconstruction = decoder.get_reconstruction()

            # - - - - - - -
            # estimate back z_sample from the reconstructed image (only anatomy may be changed, no modality factors)
            with tf.variable_scope('ModalityEncoder'):
                self.z_regress = mod_encoder.estimate_z(self.reconstruction, reuse=True)

            # - - - - - - -
            # build Segmentor
            with tf.variable_scope('Segmentor'):
                segmentor = Segmentor(self.hard_anatomy, self.n_classes, is_training=self.is_training).build()
                self.pred_mask = segmentor.get_output_mask()

        return self

    def get_soft_anatomy(self):
        return self.soft_anatomy

    def get_hard_anatomy(self):
        return self.hard_anatomy

    def get_z_distribution(self):
        return self.z_mean, self.z_logvar, self.sampled_z

    def get_z_sample_estimate(self):
        return self.z_regress

    def get_input_reconstruction(self):
        return self.reconstruction

    def get_pred_mask(self, one_hot):
        """
        Notice that the output is not necessarily either one-hot encoded, nor in the range [0, 1]. Use one-hot flag to
        obtain a segmentation mask
        :param one_hot: (bool) if true, returns one-hot segmentation mask
        :return:
        """
        if not one_hot:
            return self.pred_mask
        else:
            return tf.one_hot(tf.argmax(self.pred_mask, axis=-1), self.n_classes)

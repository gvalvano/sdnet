import tensorflow as tf

RUN_ID = 'SDNet'
data_path = './data/acdc_data'


def define_flags():
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('RUN_ID', RUN_ID, "")

    # ____________________________________________________ #
    # ========== ARCHITECTURE HYPER-PARAMETERS ========== #

    # Learning rate:
    tf.flags.DEFINE_float('lr', 1e-4, 'learning rate')

    # batch size
    tf.flags.DEFINE_integer('b_size', 4, "batch size")

    tf.flags.DEFINE_integer('n_anatomical_masks', 8, "number of extracted anatomical masks")
    tf.flags.DEFINE_integer('nz_latent', 8, "number latent variable for z code (encoder modality)")

    # ____________________________________________________ #
    # =============== TRAINING STRATEGY ================== #

    tf.flags.DEFINE_bool('augment', True, "Perform data augmentation")
    tf.flags.DEFINE_bool('standardize', False, "Perform data standardization (z-score)")  # data already pre-processed

    # ____________________________________________________ #
    # =============== INTERNAL VARIABLES ================= #

    # internal variables:
    tf.flags.DEFINE_integer('num_threads', 20, "number of CPU threads for loading and pre-process data")
    tf.flags.DEFINE_integer('skip_step', 2000, "frequency of printing batch report")
    tf.flags.DEFINE_bool('tensorboard_verbose', True, "if True: save also layers weights every N epochs")

    # ____________________________________________________ #
    # ===================== DATA SET ====================== #

    # ACDC data set:
    tf.flags.DEFINE_string('acdc_data_path', data_path, """Path of data files.""")
    tf.flags.DEFINE_integer('acdc_n_train', 70, """Number of subjects for train (tot. number = 100).""")
    tf.flags.DEFINE_integer('acdc_n_valid', 10, """Number of subjects for validation (tot. number = 100).""")
    tf.flags.DEFINE_integer('acdc_n_test', 20, """Number of subjects for test (tot. number = 100).""")

    # data specs:
    tf.flags.DEFINE_list('input_size', [128, 128], "input size")
    tf.flags.DEFINE_integer('n_classes', 4, "number of classes")

    return FLAGS

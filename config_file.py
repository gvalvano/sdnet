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
    tf.flags.DEFINE_integer('num_threads', 20, "number of threads for loading data")
    tf.flags.DEFINE_integer('skip_step', 4000, "frequency of batch report prints")
    tf.flags.DEFINE_bool('tensorboard_verbose', True, "if True: save also layers weights every N epochs")

    # ____________________________________________________ #
    # ===================== DATA SET ====================== #

    # ACDC data set:
    tf.flags.DEFINE_string('acdc_data_path', data_path, """Path of data files.""")

    # data specs:
    tf.flags.DEFINE_list('input_size', [128, 128], "input size")
    tf.flags.DEFINE_integer('n_classes', 4, "number of classes")

    return FLAGS

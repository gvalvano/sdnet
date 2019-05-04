import os
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.makedirs(path)
    except OSError:
        pass


def print_yellow_text(text, sep=True):
    """ useful for debug """
    if sep:
        print('_' * 40)  # line separator
    print('\033[1;33m{0}\033[0m'.format(text))


def create_gif(gif_name, path, ext='.png', delay=30, loop=0):
    """ Create gif from the list of images under the given path
    On Mac OS X, it requires "brew install ImageMagick".
    """
    if path[-1] == '/':
        path = path[:-1]
    if gif_name[-4:] == '.gif':
        gif_name = gif_name[:-4]

    cmd = 'convert -delay {0} -loop {1} {2}/*{3} {2}/{4}.gif'.format(delay, loop, path, ext, gif_name)
    os.system(cmd)
    print('gif successfully saved as {0}.gif'.format(os.path.join(path, gif_name)))

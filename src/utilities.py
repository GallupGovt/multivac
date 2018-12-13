import os


def mkdir(directory):
    """Make a directory if it doesn't already exist."""
    if not os.path.exists(directory):
        print('Make directory %s' % directory)
        os.makedirs(directory)
    else:
        print('Directory %s already exists' % directory)
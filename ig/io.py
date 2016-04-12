import numpy as np
import sys, getopt
import os
import scipy.ndimage
import csv

def load_voxels_binary(fname, width, height, depth, max_value=255.0, zoom = 1, order = 1):
    data = np.fromfile(fname, dtype='uint8')
    voxels = np.reshape(data, (width, height, depth))/float(max_value)
    if zoom == 1:
        return voxels
    else:
        return scipy.ndimage.zoom(voxels, zoom, order = order)

def save_dict_csv(fname, params):
    f = open(fname, 'wb')
    writer = csv.writer(f)
    for key, value in params.items():
        writer.writerow([key, value])
    f.close()

def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directlds a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

def get_rnd_voxels(n):
    files = filter(lambda x:x.endswith(".raw") and "train" in x, get_filepaths(os.getenv('HOME') + '/data/ModelNet40'))
    return np.random.choice(files, n, replace=False)

def handle_args(argv):
    options = {'params_file' : '', 'learning_rate' : 0.1, 'momentum' : 0.9, 'load_params' : False}

    try:
        opts, args = getopt.getopt(argv,"hp:l:",["params_file=, learning_rate="])
    except getopt.GetoptError:
        print 'cold2.py -p <paramfile> -l <learning_rate> -m <momentum>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print 'cold2.py -p <paramfile> -p'
            sys.exit()
        elif opt in ("-p", "--params_file"):
            options['params_file'] = arg
            options['load_params'] = True
        elif opt in ("-l", "--learning_rate"):
            options['learning_rate'] = float(arg)
        elif opt in ("-m", "--momentum"):
            options['momentum'] = float(arg)

    print options
    return options

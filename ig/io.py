import numpy as np
import sys, getopt
import os

def load_voxels_binary(fname, width, height, depth, max_value=255.0):
    data = np.fromfile(fname, dtype='uint8')
    return np.reshape(data, (width, height, depth))/float(max_value)

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
    params_file = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hp:",["params_file="])
    except getopt.GetoptError:
        print 'cold2.py -p <paramfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'cold2.py -p <paramfile>'
            sys.exit()
        elif opt in ("-p", "--params_file"):
            params_file = arg
    print "Param File is: ", params_file
    return {'params_file' : params_file}

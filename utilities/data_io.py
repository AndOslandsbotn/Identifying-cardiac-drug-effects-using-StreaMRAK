import os
import numpy as np
from pathlib import Path
import json

#def load_beelerReuter(filename):
#    path = os.path.join('Data', filename)
#    npzfile = np.load(path + '.npz')
#    voltage = npzfile[npzfile.files[0]]
#    param = npzfile[npzfile.files[1]]
#    return voltage, param

def write_json(path, data):
    # Serializing json
    json_object = json.dumps(data, indent=4)

    # Writing to path
    with open(path, "w") as outfile:
        outfile.write(json_object)

def load_json_data(path):
    with open(path, "r") as f:
        return json.loads(f.read())

def load_npz_data(folder, filename):
    """Load data stored as npz file"""
    path = os.path.join(folder, filename)
    npzfile = np.load(path, allow_pickle=True)
    return npzfile


"""
Helper functions for data set tasks
"""

import gzip
import json
from pathlib import Path
from typing import Union
import numpy as np


# Functions
def load_json(filename: Union[str, Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret

def pixelToCoordinate(point, camera_params=None):
    if camera_params is None:
        fx = fy = 64.051
        c = 128.0/2
    else:
        raise(NotImplementedError("Undefined Camera Parameters"))
    
    u, v, d = point

    X = float((u - c) * d / fx)
    Y = float((v - c) * d /fy)
    Z = float(d)
    return np.array([X, Y, Z])
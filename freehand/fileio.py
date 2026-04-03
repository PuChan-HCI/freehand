import json

import numpy as np


def read_json_points(json_filename):
    """Load saved frame corner points from JSON into the project tensor layout.

    The JSON files used by the evaluation scripts store one list per frame. This
    helper converts that layout into a NumPy array with shape
    ``[xyz, corner_index, frame_index]`` so it matches the geometry utilities in
    ``freehand.metric`` and the plotting code.
    """
    with open(json_filename, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    # Reorder axes from frame-major JSON storage to the internal convention.
    return np.array(obj).transpose([1,2,0]).astype(np.float16)

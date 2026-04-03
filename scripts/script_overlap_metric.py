
import os

from freehand.fileio import read_json_points
from freehand.metric import frame_volume_overlap

# load ground-truth and predicted point sets
foldername = os.path.join(os.path.expanduser('~/'),'Scratch/overlap/testing_val_results')
# These JSON files are expected to contain per-frame corner points with shape
# [xyz, corner_index, frame_index].
ps_true = read_json_points(os.path.join(foldername,'y_actual_overlap_LH_Para_S_0000.json'))
ps_pred = read_json_points(os.path.join(foldername,'y_predicted_overlap_LH_Para_S_0000.json'))

# Compute the volumetric Dice score on a selected temporal subset. The current
# slices compare the first 100 ground-truth frames against the first 50
# predicted frames, which is useful for quick experiments but assumes those
# subsets are the intended overlap region.
DSC = frame_volume_overlap(ps_true[...,:100], ps_pred[...,:50])

from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER.HEAD_LR_FACTOR = 1.0

# ---------------------------------------------------------------------------- #
# Few shot setting
# ---------------------------------------------------------------------------- #
_C.INPUT.FS = CN()
_C.INPUT.FS.FEW_SHOT = False
_C.INPUT.FS.SUPPORT_WAY = 2
_C.INPUT.FS.SUPPORT_SHOT = 10

# ---------------------------------------------------------------------------- #
# SupCon and Crop
# ---------------------------------------------------------------------------- #
_C.OURS = CN() # by Zhiyuan Ma
_C.OURS.SUPPORT_FEATURE_ON = False
_C.OURS.GT_FEATURE_ON = False
_C.OURS.PROPOSAL_FEATURE_ON = False
_C.OURS.MASK_ON = False

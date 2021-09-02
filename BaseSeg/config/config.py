from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.CUDA = True
_C.ENVIRONMENT.SEED = 1
_C.ENVIRONMENT.NUM_GPU = 4
_C.ENVIRONMENT.RANK = '0'
_C.ENVIRONMENT.MONITOR_TIME_INTERVAL = 0.1
_C.ENVIRONMENT.DATA_BASE_DIR = None
_C.ENVIRONMENT.PHASE = 'train'
_C.ENVIRONMENT.INIT_METHOD = 'tcp://127.0.0.1:23456'
_C.ENVIRONMENT.IS_SMOKE_TEST = False
_C.ENVIRONMENT.EXPERIMENT_NAME = 'experiment'


# -----------------------------------------------------------------------------
# DataPrepare
# -----------------------------------------------------------------------------
_C.DATA_PREPARE = CN()
_C.DATA_PREPARE.TRAIN_SERIES_IDS_TXT = None
_C.DATA_PREPARE.TEST_SERIES_IDS_TXT = None
_C.DATA_PREPARE.BAD_CASE_SERIES_IDS_TXT = None
_C.DATA_PREPARE.TRAIN_IMAGE_DIR = None
_C.DATA_PREPARE.TRAIN_MASK_DIR = None
_C.DATA_PREPARE.TEST_IMAGE_DIR = None
_C.DATA_PREPARE.TEST_MASK_DIR = None
_C.DATA_PREPARE.DEFAULT_TRAIN_DB = None
_C.DATA_PREPARE.DEFAULT_VAL_DB = None
_C.DATA_PREPARE.OUT_DIR = None
_C.DATA_PREPARE.MASK_LABEL = [1, 2, 3, 4]
_C.DATA_PREPARE.EXTEND_SIZE = 20
_C.DATA_PREPARE.VAL_RATIO = 0.2
_C.DATA_PREPARE.IS_SMOOTH_MASK = False
_C.DATA_PREPARE.IS_NORMALIZATION_DIRECTION = True
_C.DATA_PREPARE.OUT_COARSE_SIZE = [160, 160, 160]
_C.DATA_PREPARE.OUT_COARSE_SPACING = None
_C.DATA_PREPARE.OUT_FINE_SIZE = [192, 192, 192]
_C.DATA_PREPARE.OUT_FINE_SPACING = None
_C.DATA_PREPARE.IS_SPLIT_5FOLD = False


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CN()
_C.DATA_LOADER.TRAIN_VAL_FOLD = 1
_C.DATA_LOADER.TRAIN_DB_FILE = None
_C.DATA_LOADER.VAL_DB_FILE = None
_C.DATA_LOADER.TEST_DB_FILE = None
_C.DATA_LOADER.TEST_IMAGE_DIR = None
_C.DATA_LOADER.TEST_MASK_DIR = None
_C.DATA_LOADER.TEST_SERIES_IDS_TXT = None
_C.DATA_LOADER.BAD_CASE_SERIES_IDS_TXT = None
_C.DATA_LOADER.BAD_CASE_AUGMENT_TIMES = 2
_C.DATA_LOADER.LABEL_INDEX = None
_C.DATA_LOADER.LABEL_NUM = None
_C.DATA_LOADER.LABEL_NAME = None
_C.DATA_LOADER.IS_COARSE = True
_C.DATA_LOADER.WINDOW_LEVEL = [-325, 325]
_C.DATA_LOADER.IS_NORMALIZATION_HU = True
_C.DATA_LOADER.IS_NORMALIZATION_DIRECTION = True
_C.DATA_LOADER.EXTEND_SIZE = 20
_C.DATA_LOADER.NUM_WORKER = 2
_C.DATA_LOADER.BATCH_SIZE = 1
_C.DATA_LOADER.FIVE_FOLD_LIST = []


# -----------------------------------------------------------------------------
# DataAugment
# -----------------------------------------------------------------------------
_C.DATA_AUGMENT = CN()
_C.DATA_AUGMENT.IS_ENABLE = False
_C.DATA_AUGMENT.IS_RANDOM_FLIP = False
_C.DATA_AUGMENT.IS_RANDOM_ROTATE = False
_C.DATA_AUGMENT.IS_RANDOM_SHIFT = False
_C.DATA_AUGMENT.IS_RANDOM_CROP_TO_LABELS = False
_C.DATA_AUGMENT.IS_ELASTIC_TRANSFORM = False
_C.DATA_AUGMENT.IS_CHANGE_ROI_HU = False
_C.DATA_AUGMENT.IS_ADD_GAUSSIAN_NOISE = False
_C.DATA_AUGMENT.ROI_HU_RANGE = [-50, 50]
_C.DATA_AUGMENT.ROTATE_ANGLE = [-10, 10]
_C.DATA_AUGMENT.MAX_EXTEND_SIZE = 30
_C.DATA_AUGMENT.SHIFT_MAX_RATIO = 0.2

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# coarse model config
_C.COARSE_MODEL = CN()
_C.COARSE_MODEL.META_ARCHITECTURE = 'UNet'
_C.COARSE_MODEL.DEEP_SUPERVISION = False
_C.COARSE_MODEL.AUXILIARY_TASK = False
_C.COARSE_MODEL.AUXILIARY_CLASS = 1
_C.COARSE_MODEL.INPUT_SIZE = [160, 160, 160]
_C.COARSE_MODEL.NUM_CLASSES = None
_C.COARSE_MODEL.NUM_CHANNELS = [8, 16, 32, 64, 128]
_C.COARSE_MODEL.NUM_BLOCKS = [2, 2, 2, 2]
_C.COARSE_MODEL.DECODER_NUM_BLOCK = 1
_C.COARSE_MODEL.NUM_DEPTH = 4
_C.COARSE_MODEL.ENCODER_CONV_BLOCK = 'ResFourLayerConvBlock'
_C.COARSE_MODEL.DECODER_CONV_BLOCK = 'ResTwoLayerConvBlock'
_C.COARSE_MODEL.CONTEXT_BLOCK = None
_C.COARSE_MODEL.IS_DEEP_SUPERVISION = False
_C.COARSE_MODEL.WEIGHT_DIR = None
_C.COARSE_MODEL.IS_PREPROCESS = False
_C.COARSE_MODEL.IS_POSTPROCESS = False
_C.COARSE_MODEL.IS_DYNAMIC_EMPTY_CACHE = False

# -----------------------------------------------------------------------------
# fine model config
_C.FINE_MODEL = CN()
_C.FINE_MODEL.META_ARCHITECTURE = 'UNet'
_C.FINE_MODEL.DEEP_SUPERVISION = False
_C.FINE_MODEL.AUXILIARY_TASK = False
_C.FINE_MODEL.AUXILIARY_CLASS = 1
_C.FINE_MODEL.INPUT_SIZE = [192, 192, 192]
_C.FINE_MODEL.NUM_CLASSES = None
_C.FINE_MODEL.NUM_CHANNELS = [16, 32, 64, 128, 256]
_C.FINE_MODEL.NUM_BLOCKS = [2, 2, 2, 2]
_C.FINE_MODEL.DECODER_NUM_BLOCK = 2
_C.FINE_MODEL.NUM_DEPTH = 4
_C.FINE_MODEL.ENCODER_CONV_BLOCK = 'ResFourLayerConvBlock'
_C.FINE_MODEL.DECODER_CONV_BLOCK = 'ResTwoLayerConvBlock'
_C.FINE_MODEL.CONTEXT_BLOCK = None
_C.FINE_MODEL.IS_DEEP_SUPERVISION = False
_C.FINE_MODEL.WEIGHT_DIR = None
_C.FINE_MODEL.IS_PREPROCESS = True
_C.FINE_MODEL.IS_POSTPROCESS = True
_C.FINE_MODEL.IS_DYNAMIC_EMPTY_CACHE = False

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAINING = CN()
_C.TRAINING.IS_DISTRIBUTED_TRAIN = True
_C.TRAINING.IS_APEX_TRAIN = True
_C.TRAINING.ACTIVATION = 'sigmoid'
_C.TRAINING.LOSS = 'dice'
_C.TRAINING.METRIC = 'dice'

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.TRAINING.OPTIMIZER = CN()
_C.TRAINING.OPTIMIZER.METHOD = 'adam'
_C.TRAINING.OPTIMIZER.LR = 1e-3
_C.TRAINING.OPTIMIZER.L2_PENALTY = 5e-5

# -----------------------------------------------------------------------------
# Saver
# -----------------------------------------------------------------------------
_C.TRAINING.SAVER = CN()
_C.TRAINING.SAVER.SAVER_DIR = './output'
_C.TRAINING.SAVER.SAVER_FREQUENCY = 5  # batches to wait before logging train status

# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------
_C.TRAINING.SCHEDULER = CN()
_C.TRAINING.SCHEDULER.START_EPOCH = 0
_C.TRAINING.SCHEDULER.TOTAL_EPOCHS = 60
_C.TRAINING.SCHEDULER.LR_SCHEDULE = 'cosineLR'

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TESTING = CN()
_C.TESTING.COARSE_MODEL_WEIGHT_DIR = None
_C.TESTING.FINE_MODEL_WEIGHT_DIR = None
_C.TESTING.BATCH_SIZE = 1
_C.TESTING.NUM_WORKER = 3
_C.TESTING.IS_FP16 = True
_C.TESTING.SAVER_DIR = None
_C.TESTING.IS_SAVE_MASK = False
_C.TESTING.IS_POST_PROCESS = False
_C.TESTING.IS_SYNCHRONIZATION = False
_C.TESTING.OUT_RESAMPLE_MODE = 3


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

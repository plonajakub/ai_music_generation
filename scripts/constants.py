PATH_TO_RAW_MIDIS = '..\\data\\bach_all_not_corrupted\\bach_all_not_corrupted_data'
PATH_TO_TRANSLATED_DATASETS = '..\\data\\pickles'
PATH_TO_COMPOSITIONS = '..\\compositions'
PATH_TO_CHECKPOINTS = '..\\training_checkpoints'

CHECKPOINT_NAME_GLOB = 'ckpt*'
CHECKPOINT_NAME_FORMAT = 'ckpt'

FIELD_SEPARATOR = '$'

# FN - file name
FN_TRANSLATED_MIDIS = 'translated_midis.pickle'
FN_NOTE2IDX = 'note2idx.pickle'
FN_IDX2NOTE = 'idx2note.pickle'
FN_TRANSLATED_MIDIS_PARAMS = 'params_py_dict.pickle'
FN_MODEL_PARAMS = 'model_params.pickle'
FN_PROCESSED_FILES = 'processed_files.pickle'
FN_MODEL_LOSS = 'model_loss.pickle'

# PM - parameter
PM_WITH_TIMING = 'WITH_TIMING'
PM_BATCH_SIZE = 'BATCH_SIZE'
PM_EMBEDDING_DIM = 'EMBEDDING_DIM'
PM_RNN_UNITS = 'RNN_UNITS'
PM_RNN_STATEFUL = 'RNN_STATEFUL'
PM_DATASET_FLAT = 'DATASET_FLAT'
PM_SEQ_LEN = 'SEQ_LEN'
PM_TRANSLATED_DATASET_NAME = 'TRANSLATED_DATASET_NAME'

LINE_SEPARATOR = '=' * 50
LINE_SEPARATOR_2 = '-' * 50

TRAIN_REPORT_FREQ = 10  # Number of batches between reports

# Paths
CSV_PATH= 'dataset/data.csv'
CSV_TRAIN_PATH= 'dataset/df_train.csv'
CSV_VAL_PATH= 'dataset/df_val.csv'
CSV_TEST_PATH= 'dataset/df_test.csv'
IMG_PATHS = 'dataset/images'

# Pre Process CSV
FIELDS= ['Image Index', 'Finding Labels', 'Patient ID']
# EXCLUDE_LABELS = ['No Finding', 'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Mass', 'Nodule', 'Atelectasis', 'Fibrosis', 'Edema', 'Consolidation', 'Infiltration']
EXCLUDE_LABELS = ['No Finding']
# Image Pre Process
WIDTH= 256
HEIGHT= 256

# Dataloaders
BATCH_SIZE= 32

# Random
SEED= 123
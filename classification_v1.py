from utils import *
from configs import *
from dataloader import *
from classification_model_v1.densenet_model import *

X, y= read_dataset()

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.1, validation_size=0.2, random_state=None)
num_classes, names = get_unique_labels(y)

train_dataloader= Chest_DataLoader(X_train, y_train, BATCH_SIZE, num_classes, names)
val_dataloder= Chest_DataLoader(X_val, y_val, BATCH_SIZE, num_classes, names)
test_dataloder= Chest_DataLoader(X_test, y_test, BATCH_SIZE, num_classes, names)

classification_model = classification_model_v1((256, 256, 3), num_classes)

early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=7,
        restore_best_weights=True
    )

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode='min',
    factor      =   .1,
    patience    =   3,
    min_lr      =   0.000001,
    min_delta   =   0.001
)

classification_model.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True),  
                            metrics=[tf.keras.metrics.AUC(multi_label=True),
                                     tf.keras.metrics.Recall(thresholds=0.5),
                                     tf.keras.metrics.Precision(thresholds=0.5)])

history = classification_model.fit(train_dataloader, 
        validation_data = val_dataloder,
        epochs = 15,
        callbacks=[
            lr_scheduler,
            early_stopping
            ])


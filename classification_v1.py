from utils import *
from configs import *
from dataloader import *
from classification_model_v1.densenet_model import *

X, y= read_dataset()

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.1, validation_size=0.2, random_state=SEED)
num_classes, names = get_unique_labels(y)

print(f'Size X_train: {len(X_train)} Pacients')
print(f'Size X_val: {len(X_val)} Pacients')
print(f'Size X_test: {len(X_test)} Pacients')

train_dataloader= Chest_DataLoader(X_train, y_train, BATCH_SIZE, num_classes, names)
val_dataloder= Chest_DataLoader(X_val, y_val, BATCH_SIZE, num_classes, names)
test_dataloder= Chest_DataLoader(X_test, y_test, 1, num_classes, names)

# Debug to see if there is any intersection between the 3 sets
# validate_dataloaders(train_dataloader, val_dataloder, test_dataloder)

# v1 or v2 models
classification_model = classification_model_v2((256, 256, 3), num_classes)
# classification_model = classification_model_v1((256, 256, 3), num_classes)

early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=6,
        restore_best_weights=True
    )

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode='min',
    factor      =   .1,
    patience    =   3,
    min_lr      =   0.0000001,
    min_delta   =   0.001
)

classification_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                            loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True),  
                            metrics=[tf.keras.metrics.AUC(multi_label=True, curve='ROC')])



history = classification_model.fit(
        train_dataloader, 
        validation_data= val_dataloder,
        epochs= 1,
        callbacks=[
            lr_scheduler,
            early_stopping
            ])

# # classification_model= tf.keras.models.load_model('/home/matheus_levy/workspace/lucas/metrics/model_v1/model.hdf5')

print('Evaluate Model')
classification_model.evaluate(test_dataloder, verbose=1)

classification_model.save(filepath=f"/home/matheus_levy/workspace/lucas/metrics/model_v2/model_v2.hdf5")
save_history(history.history, f"/home/matheus_levy/workspace/lucas/metrics/model_v2", branch="global")
   
predictions= classification_model.predict(test_dataloder, verbose=1)
results_global = evaluate_classification_model(
        test_dataloder.get_labels(), predictions, test_dataloder.labels_name)

store_test_metrics(results_global, path='/home/matheus_levy/workspace/lucas/metrics/model_v2',
                             filename=f"metrics_global_v2", name='classification_model_v2', json=True)
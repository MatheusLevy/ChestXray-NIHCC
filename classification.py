from utils import *
from configs import *
from dataloader import *
from classification_model_v1.densenet_model import *
from classification_model_v1.fusion_model import *
from classification_model_v1.convmixer import *
from tensorflow.keras.utils import plot_model
import optuna
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


X,y = read_dataset(CSV_PATH)
X_train, y_train= read_dataset(CSV_TRAIN_PATH)
X_val, y_val= read_dataset(CSV_VAL_PATH)
X_test, y_test= read_dataset(CSV_TEST_PATH)

num_classes, names = get_unique_labels(y)
print(num_classes)
print(names)
print(f'Size X_train: {len(X_train)} Pacients')
print(f'Size X_val: {len(X_val)} Pacients')
print(f'Size X_test: {len(X_test)} Pacients')

train_dataloader= Chest_DataLoader(X_train, y_train, BATCH_SIZE, num_classes, names)
val_dataloder= Chest_DataLoader(X_val, y_val, BATCH_SIZE, num_classes, names)
test_dataloder= Chest_DataLoader(X_test, y_test, 1, num_classes, names)

#  Debug to see if there is any intersection between the 3 sets
validate_dataloaders(train_dataloader, val_dataloder, test_dataloder)

# New Dataloader
df_train = pd.read_csv(CSV_TRAIN_PATH)
# df_train = df_train_full[df_train_full['Finding Labels'].str.contains(er)]
# filtered_df = df_train_full[~df_train_full['Finding Labels'].str.contains(er)]
# random_samples = filtered_df.sample(n=16000, replace=False)
# df_train = df_train.append(random_samples, ignore_index=True)

print("Train")
count_classes(df_train, names)

df_val = pd.read_csv(CSV_VAL_PATH)
print("Val")
count_classes(df_val, names)

df_test = pd.read_csv(CSV_TEST_PATH)
print("Teste")
count_classes(df_test, names)


# generato
train_generator = get_generator_train(
    df=df_train, x_col="path", names=names, shuffle=True, batch_size=BATCH_SIZE)
val_generator = get_generator_val(
    df=df_val, x_col="path", names=names, shuffle=False)
test_generator = get_generator_val(
    df=df_test, x_col="path", names=names, shuffle=False)


fusion_model = fusion_model((256, 256, 3), num_classes)
fusion_model.summary()
# densenet_pneumo = tf.keras.models.load_model('/home/matheus_levy/workspace/lucas/metrics/densenet121_chest4_no_top')
# print(fusion_model_.summary())

# # conv1_1 = fusion_model_.get_layer('densenet121').get_layer('conv1/conv_2').get_weights()[0]
# # conv1_2 = densenet_pneumo.get_layer('conv1/conv').get_weights()[0]
# # print(np.array_equal(conv1_1, conv1_2))

early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=15,
        restore_best_weights=True
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode='min',
    factor      =   .1,
    patience    =   4,
    min_lr      =   0.00000000001,
    min_delta   =   0.001
)

adam_opt = tf.keras.optimizers.Adamax(learning_rate = 1e-3)
fusion_model.compile(optimizer=adam_opt,
                            loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, gamma=4, alpha=0.8),
                            metrics=[tf.keras.metrics.AUC(multi_label=True, curve='ROC')])

history = fusion_model.fit(
        train_generator, 
        validation_data=val_generator,
        epochs= 50,
        callbacks=[
            early_stopping,
            lr_scheduler
            ])

# densenet_pneumo.layers[0].save('densenet121_pneumo_no-top')
fusion_model.save('fusion_model_v1')
# fusion_model_.save('densenet121_pneumo_with_top')

# classification_model= tf.keras.models.load_model('/home/matheus_levy/workspace/lucas/metrics/model_v1/model.hdf5')
    
print('Evaluating Model...')
fusion_model.evaluate(test_generator, verbose=1)

print('Saving Model...')
fusion_model.save(filepath=f"/home/matheus_levy/workspace/lucas/metrics/model_v4/model_v4")
save_history(history.history, f"/home/matheus_levy/workspace/lucas/metrics/model_v4", branch="global")

print('Saving Metrics...')
predictions= fusion_model.predict(test_generator, verbose=1)
results_global = evaluate_classification_model(
        test_generator.labels, predictions, names)


store_test_metrics(results_global, path='/home/matheus_levy/workspace/lucas/metrics/model_v4',
                            filename=f"metrics_global_v4", name='classification_model_v4', json=True, result_path='/home/matheus_levy/workspace/lucas/metrics/model_v4/result_v4.json')



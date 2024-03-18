from utils import *
from configs import *
from dataloader import *
from classification_model_v1.densenet_model import *
from classification_model_v1.convmixer import *
from tensorflow.keras.utils import plot_model
import optuna
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

X,y = read_dataset(CSV_PATH)
# X_train, y_train= read_dataset(CSV_TRAIN_PATH)
# X_val, y_val= read_dataset(CSV_VAL_PATH)
# X_test, y_test= read_dataset(CSV_TEST_PATH)

num_classes, names = get_unique_labels(y)

# print(f'Size X_train: {len(X_train)} Pacients')
# print(f'Size X_val: {len(X_val)} Pacients')
# print(f'Size X_test: {len(X_test)} Pacients')

# train_dataloader= Chest_DataLoader(X_train, y_train, BATCH_SIZE, num_classes, names)
# val_dataloder= Chest_DataLoader(X_val, y_val, BATCH_SIZE, num_classes, names)
# test_dataloder= Chest_DataLoader(X_test, y_test, 1, num_classes, names)

# # Debug to see if there is any intersection between the 3 sets
# validate_dataloaders(train_dataloader, val_dataloder, test_dataloder)

# New Dataloader
df_train = pd.read_csv(CSV_TRAIN_PATH)
df_val = pd.read_csv(CSV_VAL_PATH)
df_test = pd.read_csv(CSV_TEST_PATH)

# generato
train_generator = get_generator_train(
    df=df_train, x_col="path", names=names, shuffle=True, batch_size=BATCH_SIZE)
val_generator = get_generator_val(
    df=df_val, x_col="path", names=names, shuffle=False)
test_generator = get_generator_val(
    df=df_test, x_col="path", names=names, shuffle=False)


import gc
def objective(trial):
    gc.collect()
    hparams = {
        'Quantidade de Filtros': trial.suggest_int("filters", 128, 1536, step= 128),
        'Profundidade': trial.suggest_int("depth", 3, 32, step= 1),
        'Tamanho do Kernel': trial.suggest_int("kernel_size", 3, 9, step= 1),
        'Tamanho do Patch': trial.suggest_int("patch_size", 2, 14, step= 1)
    }
    try:
        classification_model = get_conv_mixer(image_size=256, filters=hparams['Quantidade de Filtros'], depth=hparams['Profundidade'], kernel_size=hparams['Tamanho do Kernel'], patch_size=hparams['Tamanho do Patch'], num_classes=num_classes)
        classification_model.summary()

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
            min_lr      =   0.000000001,
            min_delta   =   0.001
        )



        adam_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        classification_model.compile(optimizer=adam_opt,
                                    loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True),  
                                    metrics=[tf.keras.metrics.AUC(multi_label=True, curve='ROC')])


        with wandb.init(project='ConvMixer Otimização', name="ConvMixer_trial_{NUM}".format(NUM=trial.number), config=hparams):
            config = wandb.config
            history = classification_model.fit(
                        train_generator, 
                        validation_data=val_generator,
                        epochs= 15,
                        callbacks=[
                            early_stopping,
                            lr_scheduler
                            ])

        # classification_model= tf.keras.models.load_model('/home/matheus_levy/workspace/lucas/metrics/model_v1/model.hdf5')

        # print('Evaluating Model...')
        # classification_model.evaluate(test_generator, verbose=1)

        # print('Saving Model...')
        # classification_model.save(filepath=f"/home/matheus_levy/workspace/lucas/metrics/model_v2/model_v2.hdf5")
        # save_history(history.history, f"/home/matheus_levy/workspace/lucas/metrics/model_v2", branch="global")
        
        # print('Saving Metrics...')
        # predictions= classification_model.predict(test_generator, verbose=1)
        # results_global = evaluate_classification_model(
        #         test_generator.labels, predictions, names)


        # store_test_metrics(results_global, path='/home/matheus_levy/workspace/lucas/metrics/model_v2',
        #                             filename=f"metrics_global_v2", name='classification_model_v2', json=True, result_path='/home/matheus_levy/workspace/lucas/metrics/model_v2/result_v2.json')

        # print('Training End')
            auc_test= classification_model.evaluate(test_generator, verbose=1)

            if (trial.number > 0):
                wandb.log({
                    'train_loss':  history.history['loss'][-1],
                    'train_auc':  history.history[f'auc_{trial.number}'][-1],
                    'val_loss': history.history['val_loss'][-1],
                    'val_auc': history.history[f'val_auc_{trial.number}'][-1],
                    'test_auc': auc_test[1]  
                })
            else:
                wandb.log({
                    'train_loss':  history.history['loss'][-1],
                    'train_auc':  history.history['auc'][-1],
                    'val_loss': history.history['val_loss'][-1],
                    'val_auc': history.history['val_auc'][-1],
                    'test_auc': auc_test[1]  
                })

            return auc_test[1]  
    
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        raise optuna.exceptions.TrialPruned()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=60)
print(study.best_params)

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
  print("    {}: {}".format(key, value))

wandb.finish()
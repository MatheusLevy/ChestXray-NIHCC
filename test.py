import numpy as np
from keras.utils import Sequence
import tensorflow as tf

class InferenceDataGenerator(Sequence):
    def __init__(self, x_data, batch_size):
        self.x_data = x_data
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.x_data))

    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch_x = self.x_data[start_idx:end_idx]

        # Você pode adicionar qualquer pré-processamento necessário aqui

        return batch_x

# Exemplo de uso
# Suponha que você tenha dados de teste x_test
x_test = np.random.rand(100, 64, 64, 3)

batch_size = 32
inference_data_generator = InferenceDataGenerator(x_test, batch_size)

classification_model= tf.keras.models.load_model('/home/matheus_levy/workspace/lucas/model.hdf5')
predictions = classification_model.predict(inference_data_generator, verbose=1)
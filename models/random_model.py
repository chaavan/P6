from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(1./255, input_shape=input_shape),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_normal', bias_initializer='zeros'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'),
            layers.Dropout(0.5),
            layers.Dense(categories_count, activation='softmax', kernel_initializer='random_normal', bias_initializer='zeros')
        ])
    
    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

class ConvNet:
    def get_model(self, img_height, img_width, summary=True):
        model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu',
                   input_shape=(img_height, img_width, 3)),
            MaxPooling2D(),
            Dropout(0.2),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(4, activation='softmax') #3 categories as output channel
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model


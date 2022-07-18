import tensorflow as tf

class InceptionResNetV2:
    def get_model(self, img_height, img_width, summary=True):
        base_model = tf.keras.applications.InceptionResNetV2(input_shape=(img_height, img_width, 3),
                                                       include_top=False,
                                                       weights='imagenet')

        base_model.trainable = False

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(1)

        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])

        base_learning_rate = 0.0001
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model

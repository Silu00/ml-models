import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model_emnist():
    try:
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    except Exception as e:
        print("Error: ", e)

    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.35,
        fill_mode='constant',
        cval=0
    )
    datagen.fit(train_images)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(datagen.flow(train_images, train_labels, batch_size=128), epochs=15, validation_data=(test_images, test_labels))

    model.save('model_emnist.h5')

if __name__ == "__main__":
    train_model_emnist()

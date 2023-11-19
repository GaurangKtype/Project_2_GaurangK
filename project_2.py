#Code start#

# Importing Libraries

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomZoom, RandomRotation
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Loading all the datasets

image_shape = (100, 100, 3)
train_dir = 'G:\My Drive\Gaurang Files\TMU\Year 4\AER 850 Intro to Machine Learning\Project\Project_2_GaurangK\Data\Train'
val_dir = 'G:\My Drive\Gaurang Files\TMU\Year 4\AER 850 Intro to Machine Learning\Project\Project_2_GaurangK\Data\Validation'
test_dir = 'G:\My Drive\Gaurang Files\TMU\Year 4\AER 850 Intro to Machine Learning\Project\Project_2_GaurangK\Data\Test'

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size = (100, 100),
    label_mode = 'categorical'
    )
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size = (100, 100),
    label_mode = 'categorical'
    )
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = (100, 100),
    label_mode = 'categorical'
    )

model_1 = models.Sequential([
    # Rescaling layer to normalize the pixel values
    layers.Rescaling(1./255, input_shape=image_shape),

    # Data augmentation layers
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),

    # First Convolutional Layer
    layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=image_shape, padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

    # Second Convolutional Layer
    layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

    # Third Convolutional Layer
    layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),

    # Fourth Convolutional Layer
    layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),

    # Fifth Convolutional Layer
    layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

    # Flattening the 3D output to 1D
    layers.Flatten(),

    # First Fully Connected Layer
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),

    # Second Fully Connected Layer
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),

    # Third Fully Connected Layer
    # Output layer with number of neurons equal to the number of classes
    layers.Dense(4, activation='softmax')
])

# Compile the model
model_1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Model summary
model_1.summary()

# Train the model using the train and validation datasets
history = model_1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
# Compile the model
model_1.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model_1.summary()

# Train the model using the train and validation datasets
history = model_1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
# Retrieve the training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Retrieve the number of epochs
epochs = range(1, len(train_loss) + 1)

# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plot
plt.show()



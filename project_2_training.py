# Importing Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

# Loading all the datasets
image_shape = (100, 100, 3)
base_dir = 'G:\\My Drive\\Gaurang Files\\TMU\\Year 4\\AER 850 Intro to Machine Learning\\Project\\Project_2_GaurangK\\Data\\'
train_dir = base_dir + 'Train'
val_dir = base_dir + 'Validation'
test_dir = base_dir + 'Test'

def load_datasets(directory, image_size=(100, 100), label_mode='categorical'):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        label_mode=label_mode
    )

train_ds = load_datasets(train_dir)
val_ds = load_datasets(val_dir)
test_ds = load_datasets(test_dir)

def build_model(image_shape):
    model = models.Sequential([
        # Rescaling layer to normalize the pixel values
        layers.Rescaling(1./255, input_shape=image_shape),
        
        # Data augmentation layers
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        
        # Convolutional and Pooling Layers
        layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=image_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        
        # Flattening the 3D output to 1D
        layers.Flatten(),
        
        # Dense Layers with L2 Regularization
        layers.Dense(4096, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(4, activation='softmax')
    ])
    
    return model

# Define the image shape and number of classes
image_shape = (100, 100, 3)
num_classes = 4

# Build the DenseNet model
model_1 = build_model(image_shape)
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_1.summary()

#model_1 = build_model(image_shape)
#model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model_1.summary()

# Training the model
history = model_1.fit(train_ds, validation_data=val_ds, epochs=10)

# Plotting the training and validation loss
def plot_loss_accuracy(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_loss_accuracy(history)

# Evaluate the model on the test set
test_loss, test_accuracy = model_1.evaluate(test_ds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

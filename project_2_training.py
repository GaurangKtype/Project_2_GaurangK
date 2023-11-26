# Importing Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define image shape and directory paths
image_shape = (100, 100, 3)
base_dir = 'G:\\My Drive\\Gaurang Files\\TMU\\Year 4\\AER 850 Intro to Machine Learning\\Project\\Project_2_GaurangK\\Data\\'
train_dir = base_dir + 'Train'
val_dir = base_dir + 'Validation'
test_dir = base_dir + 'Test'

# Define ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Connect the ImageDataGenerators to the respective directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_shape[:2],
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_shape[:2],
    batch_size=32,
    class_mode='categorical'
)

# Test data should not be augmented, just rescaled
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_shape[:2],
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Usually, we don't shuffle test data
)

# Model building function without the internal data augmentation layers
def build_model(image_shape):
    model = models.Sequential([  
        # Convolutional and Pooling Layers
        layers.Conv2D(64, (11, 11), strides=(4, 4), activation='relu', input_shape=image_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(128, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        
        # Flattening the 3D output to 1D
        layers.Flatten(),
        
        # Dense Layers with L2 Regularization
        layers.Dense(2048, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
     
        # Output Layer
        layers.Dense(4, activation='softmax')
    ])
    
    return model

image_shape = (100, 100, 3)
num_classes = 4

# Assuming build_model is defined as in the previous message

# Build the DenseNet model
model_1 = build_model(image_shape)

# Specify the learning rate
learning_rate = 0.01  # You can adjust this value

# Create an Adam optimizer with the given learning rate
adam_optimizer = Adam(learning_rate=learning_rate)
        
# Compile the model
model_1.compile(optimizer=adam_optimizer, 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

# Model summary
model_1.summary()

# Training the model using the data generators
history = model_1.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//validation_generator.batch_size,
    epochs=5
)

# Save the model
model_save_path = 'G:\My Drive\Gaurang Files\TMU\Year 4\AER 850 Intro to Machine Learning\Project\Project_2_GaurangK\Model_100_epoch'  
model_1.save(model_save_path)

# Plotting the training and validation loss and accuracy
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

# Evaluate the model on the test set using the test generator
test_loss, test_accuracy = model_1.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
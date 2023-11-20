# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:09:01 2023

@author: gaura
"""

## Models

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
        layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
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
##################################################################################
# Define the model architecture
def build_model(image_shape):
    model = models.Sequential([
        # Rescaling layer to normalize the pixel values
        layers.Rescaling(1./255, input_shape=image_shape),
        
        # Data augmentation layers
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        
        # Convolutional and Pooling Layers
        layers.Conv2D(48, (11, 11), strides=(4, 4), activation='relu', input_shape=image_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(128, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(192, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        
        # Flattening the 3D output to 1D
        layers.Flatten(),
        
        # Dense Layers with L2 Regularization
        layers.Dense(2048, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(2048, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(4, activation='softmax')
    ])
    
    return model
##################################################################################
def load_datasets(directory, image_size=(100, 100), label_mode='categorical'):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        label_mode=label_mode
    )

train_ds = load_datasets(train_dir)
val_ds = load_datasets(val_dir)
test_ds = load_datasets(test_dir)

def dense_block(x, blocks, growth_rate):
    for i in range(blocks):
        x = conv_block(x, growth_rate)
    return x

def conv_block(x, growth_rate):
    x1 = layers.BatchNormalization()(x)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, (1, 1), use_bias=False)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(growth_rate, (3, 3), padding='same', use_bias=False)(x1)
    x = layers.Concatenate()([x, x1])
    return x

def transition_layer(x, reduction):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(int(tf.keras.backend.int_shape(x)[-1] * reduction), (1, 1), use_bias=False)(x)
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

def build_densenet_model(image_shape, num_classes):
    growth_rate = 32
    num_blocks = 3
    num_layers_per_block = 4
    reduction = 0.5

    input_img = layers.Input(shape=image_shape)

    # Initial convolution
    x = layers.Conv2D(2 * growth_rate, (7, 7), strides=(2, 2), padding='same', use_bias=False)(input_img)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Dense blocks and transition layers
    for i in range(num_blocks):
        x = dense_block(x, num_layers_per_block, growth_rate)
        if i != num_blocks - 1:  # no transition layer after the last dense block
            x = transition_layer(x, reduction)

    # Final batch norm and relu
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Global Average Pooling and output layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)

    model = models.Model(inputs=input_img, outputs=x)
    
    return model
##############################################################################

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
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
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
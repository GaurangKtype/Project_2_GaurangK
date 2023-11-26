## Test File - Step 5##
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path, target_size=(100, 100)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Normalize the image array
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    return img_array

# Function to display image and predictions
def display_predictions(image_path, model):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    
    # Display the image
    img = image.load_img(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide the axis
    
    # Get class names
    classes = ['Large Crack', 'Medium Crack', 'Small Crack', 'No Crack']
    
    # Calculate the height at which the text should start (at the bottom of the image)
    # This is based on the size of the figure, the desired font size, and the number of classes
    img_height = img.size[1]
    start_height = img_height - 10 - (15 * len(classes))
    
    # Display all predictions on the image
    for i, class_name in enumerate(classes):
        probability = predictions[0][i] * 100
        label = f"{class_name}: {probability:.2f}%"
        plt.text(10, start_height, label, color='green', fontsize=10, backgroundcolor='black')
        start_height += 160  # Increment for the next label to have space between lines

    plt.show()

model = tf.keras.models.load_model('G:\My Drive\Gaurang Files\TMU\Year 4\AER 850 Intro to Machine Learning\Project\Project_2_GaurangK\Model_100_epoch')

# Paths to your test images
test_image_paths = ['G:\My Drive\Gaurang Files\TMU\Year 4\AER 850 Intro to Machine Learning\Project\Project_2_GaurangK\Data\Test\Medium\Crack__20180419_06_19_09,915.bmp',
                    'G:\My Drive\Gaurang Files\TMU\Year 4\AER 850 Intro to Machine Learning\Project\Project_2_GaurangK\Data\Test\Large\Crack__20180419_13_29_14,846.bmp']

# Loop through the images and display predictions
for img_path in test_image_paths:
    display_predictions(img_path, model)

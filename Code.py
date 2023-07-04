#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import required packages
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Get Path to Train and Valid folders
train_path = r'C:\Users\KIIT\PlantDatasetimages\New Plant Diseases Dataset(Augmented)\New
Plant Diseases Dataset(Augmented)\train'
valid_path = r'C:\Users\KIIT\PlantDatasetimages\New Plant Diseases Dataset(Augmented)\New
Plant Diseases Dataset(Augmented)\valid'
# Get list of all subfolders for each Subset
train_dir = os.listdir(train_path)
valid_dir = os.listdir(valid_path)
# Check length of subfolders
len(train_dir), len(valid_dir)
data_dir = r'C:\Users\KIIT\PlantDatasetimages\New Plant Diseases Dataset(Augmented)\New
Plant Diseases Dataset(Augmented)'
train_dir = data_dir + r"\train"
valid_dir = data_dir + r"\valid"
diseases = os.listdir(train_dir)
# Number of images for each disease
nums = {}
for disease in diseases:
nums[disease] = len(os.listdir(train_dir + '/' + disease))
# converting the nums dictionary to pandas dataframe passing index as plant name and number
of images as column
img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["no. of images"])
img_per_class
plt.figure(figsize=(10, 10))
# Calculate the percentage of each disease class
percentages = [(num / sum(nums.values())) * 100 for num in nums.values()]
# Plot the pie chart
plt.pie(percentages, labels=diseases, autopct='%1.1f%%', startangle=90)
plt.title('Images per each class of plant disease', pad=40 )
plt.axis('equal')
# Add a legend
plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.show()
# Import required packages
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Set the seed for reproducibility
tf.random.set_seed(42)
# Get Path to Train and Valid folders
train_path = r'C:\Users\KIIT\PlantDatasetimages\New Plant Diseases Dataset(Augmented)\New
Plant Diseases Dataset(Augmented)\train'
valid_path = r'C:\Users\KIIT\PlantDatasetimages\New Plant Diseases Dataset(Augmented)\New
Plant Diseases Dataset(Augmented)\valid'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Set image size and batch size
image_size = (224, 224)
batch_size = 32
# Create ImageDataGenerator instances
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# Load and preprocess the training dataset
train_generator = train_datagen.flow_from_directory(
train_path,
target_size=image_size,
batch_size=batch_size,
class_mode='categorical',
subset='training'
)
# Load and preprocess the validation dataset
valid_generator = valid_datagen.flow_from_directory(
valid_path,
target_size=image_size,
batch_size=batch_size,
class_mode='categorical',
subset='validation'
)
# Check the number of classes
num_classes = len(train_generator.class_indices)
import tensorflow as tf
import matplotlib.pyplot as plt
# Build the CNN model
model = tf.keras.Sequential([
tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(num_classes, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])
# Set the number of epochs
num_epochs = 60
# Train the model and get training history
history = model.fit_generator(train_generator, epochs=num_epochs,
validation_data=valid_generator)
# Save the model to the desired path
save_dir = r'C:\Users\KIIT\PlantDatasetimages'
save_path = os.path.join(save_dir, 'model.h5')
model.save(save_path)
# Plot accuracy and loss curves
plt.figure(figsize=(12, 4))
# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'valid'])
# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'valid'])
plt.tight_layout()
plt.show()
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
def test_model(model_path, test_dir, download_output=False):
# Load the trained model
model = load_model(model_path)
# Get the class labels from the training generator
class_labels = list(train_generator.class_indices.keys())
# Initialize lists to store the predicted labels and corresponding image paths
predicted_labels = []
image_paths = []
# Iterate over the images in the test directory
for image_file in os.listdir(test_dir):
# Create the image file path
image_path = os.path.join(test_dir, image_file)
image_paths.append(image_path)
# Load and preprocess the image
image = load_img(image_path, target_size=image_size)
image_array = img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)
# Make predictions
prediction = model.predict(image_array)
predicted_label = class_labels[np.argmax(prediction)]
predicted_labels.append(predicted_label)
# Display the image with the predicted label
plt.figure()
plt.imshow(image)
plt.title("Predicted Label: " + predicted_label)
plt.axis('off')
# Print the predicted labels and corresponding image paths in a tabular format
print("Image\t\t\t\tPredicted Label")
print("----------------------------------------------")
for i in range(len(predicted_labels)):
print(os.path.basename(image_paths[i]), "\t\t", predicted_labels[i])
# Save the entire output as a single JPG image
if download_output:
output_image = Image.new('RGB', (600, 300 * len(predicted_labels)), color=(255, 255,
255))
y_offset = 0
for i in range(len(predicted_labels)):
image = Image.open(image_paths[i])
output_image.paste(image, (0, y_offset))
draw = ImageDraw.Draw(output_image)
draw.text((image.size[0] + 10, y_offset + 10), predicted_labels[i], fill=(0, 0, 0))
y_offset += image.size[1] + 10
output_image.save("output.jpg")
# Specify the path to the trained model and the test directory
model_path = r'C:\Users\KIIT\PlantDatasetimages\model.h5'
test_directory = r'C:\Users\KIIT\PlantDatasetimages\test\test'
# Call the test_model function
test_model(model_path, test_directory, download_output=True)
VGG16
VGG 16 was proposed by Karen Simonyan and Andrew Zisserman of the Visual Geometry Group
Lab of Oxford University in 2014 in the paper “VERY DEEP CONVOLUTIONAL NETWORKS FOR
LARGE-SCALE IMAGE RECOGNITION”. Also called VGGNet. It is a convolution neural network
(CNN) model supporting 16 layers.
# Import required packages
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Set the seed for reproducibility
tf.random.set_seed(42)
# Set the paths to train and valid folders
train_path = r'C:\Users\KIIT\PlantDatasetimages\New Plant Diseases Dataset(Augmented)\New
Plant Diseases Dataset(Augmented)\train'
valid_path = r'C:\Users\KIIT\PlantDatasetimages\New Plant Diseases Dataset(Augmented)\New
Plant Diseases Dataset(Augmented)\valid'
# Get list of all subfolders for each subset
train_dir = os.listdir(train_path)
valid_dir = os.listdir(valid_path)
print(len(train_dir), len(valid_dir))
# Set the image size and batch size
image_size = (224, 224)
batch_size = 32
# Create ImageDataGenerator instances with data augmentation and normalization
train_datagen = ImageDataGenerator(
rescale=1./255,
validation_split=0.2,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# Load and preprocess the training dataset
train_generator = train_datagen.flow_from_directory(
train_path,
target_size=image_size,
batch_size=batch_size,
class_mode='categorical',
subset='training'
)
# Load and preprocess the validation dataset
valid_generator = valid_datagen.flow_from_directory(
valid_path,
target_size=image_size,
batch_size=batch_size,
class_mode='categorical',
subset='validation'
)
# Check the number of classes
num_classes = len(train_generator.class_indices)
from tensorflow.keras.applications import VGG16
# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the base model's layers
base_model.trainable = False
# Create the new model architecture by adding the top layers
model_vgg16 = tf.keras.Sequential([
base_model,
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(num_classes, activation='softmax')
])
# Compile the model
model_vgg16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model and get training history
history_vgg16 = model_vgg16.fit(train_generator, epochs=60, validation_data=valid_generator)
# Save the model
save_dir = r'C:\Users\KIIT\PlantDatasetimages'
save_path_vgg16 = os.path.join(save_dir, 'model_vgg16.h5')
model_vgg16.save(save_path_vgg16)
RESNET50
# Import required packages
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Set the seed for reproducibility
tf.random.set_seed(42)
# Set the paths to train and valid folders
train_path = r'C:\Users\KIIT\PlantDatasetimages\New Plant Diseases Dataset(Augmented)\New
Plant Diseases Dataset(Augmented)\train'
valid_path = r'C:\Users\KIIT\PlantDatasetimages\New Plant Diseases Dataset(Augmented)\New
Plant Diseases Dataset(Augmented)\valid'
# Get list of all subfolders for each subset
train_dir = os.listdir(train_path)
valid_dir = os.listdir(valid_path)
print(len(train_dir), len(valid_dir))
# Set the image size and batch size
image_size = (224, 224)
batch_size = 32
# Create ImageDataGenerator instances with data augmentation and normalization
train_datagen = ImageDataGenerator(
rescale=1./255,
validation_split=0.2,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest'
)
valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# Load and preprocess the training dataset
train_generator = train_datagen.flow_from_directory(
train_path,
target_size=image_size,
batch_size=batch_size,
class_mode='categorical',
subset='training'
)
# Load and preprocess the validation dataset
valid_generator = valid_datagen.flow_from_directory(
valid_path,
target_size=image_size,
batch_size=batch_size,
class_mode='categorical',
subset='validation'
)
# Check the number of classes
num_classes = len(train_generator.class_indices)
from tensorflow.keras.applications import ResNet50
# Load the pre-trained ResNet50 model without the top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the base model's layers
base_model.trainable = False
# Create the new model architecture by adding the top layers
model_resnet50 = tf.keras.Sequential([
base_model,
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(num_classes, activation='softmax')
])
# Compile the model
model_resnet50.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
# Train the model and get training history
history_resnet50 = model_resnet50.fit(train_generator, epochs=num_epochs,
validation_data=valid_generator)
# Save the model
save_dir = r'C:\Users\KIIT\PlantDatasetimages'
save_path_resnet50 = os.path.join(save_dir, 'model_resnet50.h5')
model_resnet50.save(save_path_resnet50)
# Compare with other models
# Load and compile other models
model_2 = tf.keras.models.load_model('path_to_model_2.h5')
model_3 = tf.keras.models.load_model('path_to_model_3.h5')
# Train and get histories for other models history_2 = model_2.fit(train_generator,
epochs=num_epochs, validation_data=valid_generator) history_3 = model_3.fit(train_generator,
epochs=num_epochs, validation_data=valid_generator) # Plot validation accuracies
plt.plot(history.history['val_accuracy']) plt.plot(history_2.history['val_accuracy'])
plt.plot(history_3.history['val_accuracy']) plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs') plt.ylabel('Accuracy') plt.legend(['Model 1', 'Model 2', 'Model 3']) plt.show() #
Plot validation losses plt.plot(history.history['val_loss']) plt.plot(history_2.history['val_loss'])
plt.plot(history_3.history['val_loss']) plt.title('Validation Loss Comparison') plt.xlabel('Epochs')
plt.ylabel('Loss') plt.legend(['Model 1', 'Model 2', 'Model 3']) plt.show()


import os
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set the path to the train and test folders
train_folder = "Train"
val_folder = "Validation"
test_folder = "Test"

# Set the number of classes
num_classes = 2

# Set the input image size
input_size = (224, 224)

# Set the batch size for training and validation
batch_size = 512

# Set the number of training epochs
epochs = 25

# Data augmentation and validation split for training images
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    
)
# Data augmentation for validation and test images
val_test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load and preprocess the training data
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load and preprocess the validation data
validation_generator = val_test_datagen.flow_from_directory(
    val_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Load and preprocess the test data
test_generator = val_test_datagen.flow_from_directory(
    test_folder,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Load the pre-trained InceptionV3 model without the top classification layer
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer with 1024 units
x = Dense(1024, activation='relu')(x)

# Add an output layer with softmax activation for classification
predictions = Dense(num_classes, activation='softmax')(x)

# Create the InceptionV3 model for training
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)

# Create a DataFrame to store the training history
history_df = pd.DataFrame(history.history)

# Add test loss and accuracy to the DataFrame
history_df.loc[0, 'test_loss'] = test_loss
history_df.loc[0, 'test_accuracy'] = test_accuracy

# Print the test loss and accuracy
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Calculate the confusion matrix
# Get the predicted classes for the test images
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get the true classes from the test generator
true_classes = test_generator.classes

# Calculate the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Visualize the confusion matrix as a heatmap and save it as an image
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=test_generator.class_indices, yticklabels=test_generator.class_indices)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Calculate sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Create a DataFrame to store sensitivity and specificity
metrics_df = pd.DataFrame({
    'Metric': ['Sensitivity', 'Specificity'],
    'Value': [sensitivity, specificity]
})

# Save sensitivity and specificity to a CSV file
metrics_filename = 'metrics.csv'
metrics_df.to_csv(metrics_filename, index=False)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)

# Get the predicted classes for the test images
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1) if predictions is not None else None

# Get the class labels from the test generator
class_labels = list(test_generator.class_indices.keys())

# Print the results for each test image
print("Test Results:")
test_results = []
for i, image_path in enumerate(test_generator.filepaths):
    image_name = os.path.basename(image_path)
    true_class = class_labels[test_generator.labels[i]]
    predicted_class = class_labels[predicted_classes[i]]
    accuracy = 100 if true_class == predicted_class else 0
    print(f"Test Image Name: {image_name} - Classified as {predicted_class}/{true_class} - Accuracy: {accuracy}%")

    test_results.append({
        "Test Image Name": image_name,
        "Predicted Class": predicted_class,
        "True Class": true_class,
        "Accuracy": f"{accuracy}%"
    })

test_results_df = pd.DataFrame(test_results)

test_results_df.to_excel("test_results.xlsx", index=False)

# Save the training history to an Excel file
history_filename = f"training_history_{model.name}_{num_classes}classes.xlsx"
history_df.to_excel(history_filename, index=False)
print(f"Training history saved to '{history_filename}'")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_filename = f"model_{model.name}_{num_classes}classes.tflite"
with open(tflite_model_filename, 'wb') as f:
    f.write(tflite_model)
print("Model converted and saved as TensorFlow Lite format:", tflite_model_filename)

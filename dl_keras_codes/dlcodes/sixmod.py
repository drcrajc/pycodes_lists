import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Define the paths to the datasets
train_data_dir = 'mchipds/Train'
val_data_dir = 'mchipds/Validation'
test_data_dir = 'mchipds/Test'

# Set the image dimensions and batch size
image_height, image_width = 224, 224
batch_size = 32

# Load the training dataset
train_ds = keras.utils.image_dataset_from_directory(
    train_data_dir,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    seed=42
)

# Load the validation dataset
val_ds = keras.utils.image_dataset_from_directory(
    val_data_dir,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    seed=42
)

# Load the test dataset
test_ds = keras.utils.image_dataset_from_directory(
    test_data_dir,
    image_size=(image_height, image_width),
    batch_size=batch_size
)

# Get the class names
class_names = train_ds.class_names


def create_model(model_name):
    # Model definition
    model = Sequential()
    pretrained_model = None

    if model_name == 'Xception':
        pretrained_model = keras.applications.Xception(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(image_height, image_width, 3),
            classes=2,
            classifier_activation='softmax'
        )
    elif model_name == 'InceptionV3':
        pretrained_model = keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(image_height, image_width, 3),
            classes=2,
            classifier_activation='softmax'
        )
    elif model_name == 'MobileNet':
        pretrained_model = keras.applications.MobileNet(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(image_height, image_width, 3),
            classes=2,
            classifier_activation='softmax'
        )
    elif model_name == 'MobileNetV2':
        pretrained_model = keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(image_height, image_width, 3),
            classes=2,
            classifier_activation='softmax'
        )
    elif model_name == 'EfficientNetB4':
        pretrained_model = keras.applications.EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(image_height, image_width, 3),
            classes=2,
            classifier_activation='softmax'
        )
    elif model_name == 'EfficientNetV2S':
        pretrained_model = keras.applications.EfficientNetV2S(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(image_height, image_width, 3),
            classes=2,
            classifier_activation='softmax'
        )

    for layer in pretrained_model.layers:
        layer.trainable = False

    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model


def train_model(model, model_name):
    # Define checkpoints and early stopping
    checkpoint_path = f"mchipds/Results/chk_{model_name}_model.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_sparse_categorical_accuracy', save_best_only=True, save_weights_only=True, verbose=1, mode='auto')
    early = EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=[checkpoint, early]
    )

    # Convert the history to a DataFrame
    hist_df = pd.DataFrame(history.history)

    # Save the history to files
    hist_json_file = f"mchipds/Results/history_{model_name}.json"
    hist_csv_file = f"mchipds/Results/history_{model_name}.csv"
    hist_excel_file = f"mchipds/Results/history_{model_name}.xlsx"

    hist_df.to_json(hist_json_file)
    hist_df.to_csv(hist_csv_file)
    hist_df.to_excel(hist_excel_file)

    return model


def test_model(model, model_name):
    # Predict the test dataset
    y_pred = model.predict(test_ds)
    y_pred = np.argmax(y_pred, axis=1)

    # Get the true labels for the test dataset
    y_true = []
    for images, labels in test_ds:
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Save the confusion matrix to a CSV file
    cm_csv_file = f"mchipds/Results/confusion_matrix_{model_name}.csv"
    cm_df.to_csv(cm_csv_file, index=False)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# Define the list of models
models = ['Xception', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'EfficientNetB4', 'EfficientNetV2S']

# Train, test, and evaluate each model
for model_name in models:
    print(f"Training {model_name}...")
    model = create_model(model_name)
    model = train_model(model, model_name)
    test_model(model, model_name)

# Generate a combined confusion matrix
combined_cm = np.zeros((len(class_names), len(class_names)))

for model_name in models:
    cm_csv_file = f"mchipds/Results/confusion_matrix_{model_name}.csv"
    cm_df = pd.read_csv(cm_csv_file)
    cm = cm_df.to_numpy()
    combined_cm += cm

# Create a DataFrame for the combined confusion matrix
combined_cm_df = pd.DataFrame(combined_cm, index=class_names, columns=class_names)

# Save the combined confusion matrix to a CSV file
combined_cm_csv_file = "mchipds/Results/combined_confusion_matrix.csv"
combined_cm_df.to_csv(combined_cm_csv_file, index=False)

# Visualize the combined confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(combined_cm_df, annot=True, fmt="d", cmap="Blues")
plt.title('Combined Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

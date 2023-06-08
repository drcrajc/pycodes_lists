import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Set up the paths for the data folders
train_dir = 'Train'
valid_dir = 'Validate'
test_dir = 'Test'

# Define the image size and batch size for Xception
image_size = (299, 299)
batch_size = 32

# Create data generators for training, validation, and testing
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Create the Xception model with regularization and dropout
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define the callbacks for model checkpoint and early stopping
checkpoint_path = "chk_mobilenet_modi_model.h5"
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# Loop over the folds
k = 10
sensitivity_list = []
specificity_list = []

# Create an instance of KFold with the desired number of folds
kf = KFold(n_splits=k)

for fold, (train_index, val_index) in enumerate(kf.split(train_generator.filenames), 1):
    if 1 <= fold <= 10 or fold == 20:
        print(f"Training on fold {fold}/{k}")

        # Extract the filenames for the specific indices
        train_filenames = np.array(train_generator.filenames)[train_index]
        val_filenames = np.array(train_generator.filenames)[val_index]

        # Train the model on the training set
        train_steps = len(train_filenames) // batch_size + 1
        valid_steps = len(val_filenames) // batch_size + 1

        # Train the model on the current fold
        model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            epochs=1,
            callbacks=[checkpoint, early_stopping]
        )

        # Evaluate the model on the validation set
        print("Evaluating on validation data")
        model.load_weights(checkpoint_path)
        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        valid_predictions = model.predict(valid_generator)
        valid_predictions = (valid_predictions > 0.5).astype(int)

        # Calculate TP, TN, FP, FN
        tp = np.sum((valid_predictions == 1) & (valid_generator.classes == 1))
        tn = np.sum((valid_predictions == 0) & (valid_generator.classes == 0))
        fp = np.sum((valid_predictions == 1) & (valid_generator.classes == 0))
        fn = np.sum((valid_predictions == 0) & (valid_generator.classes == 1))

        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

# Evaluate the model on the test set
print("Evaluating on test data")
model.load_weights(checkpoint_path)
test_predictions = model.predict(test_generator)
test_predictions = (test_predictions > 0.5).astype(int)

fpr, tpr, _ = roc_curve(test_generator.classes, test_predictions)
roc_auc = auc(fpr, tpr)

# Print classification report for test set
test_report = classification_report(test_generator.classes, test_predictions, output_dict=True)
test_sensitivity = test_report.get('1', {}).get('recall', 0)
test_specificity = test_report.get('0', {}).get('recall', 0)

# Create a DataFrame for sensitivity and specificity
data = {
    'Fold': list(range(1, k+1)) + ['Test'],
    'Sensitivity': sensitivity_list + [test_sensitivity],
    'Specificity': specificity_list + [test_specificity]
}
df = pd.DataFrame(data)

# Save DataFrame to Excel file
df.to_excel('classification_metrics.xlsx', index=False)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('xcp_model.tflite', 'wb') as f:
    f.write(tflite_model)

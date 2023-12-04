import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os
import scipy

# Load labels
labels_file_path = "D:/cources/Internship/2023/Dataset/imagelabels.mat"
label_image = scipy.io.loadmat(labels_file_path)
label = label_image['labels'].reshape((-1, 1))

label -= 1  # Subtract 1 from all label values

# List all files in the image folder
image_save_folder = "D:/cources/Internship/2023/Dataset/102_flowers/jpg"
image_files = [f for f in os.listdir(image_save_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Create a mapping from image file names to labels
label_mapping = dict(zip(image_files, label))

# Split data into training and testing sets
train_paths, test_paths = train_test_split(image_files, test_size=0.2, random_state=42)


# Extract labels for training and testing
train_labels = [label_mapping[image_path] for image_path in train_paths]
test_labels = [label_mapping[image_path] for image_path in test_paths]


# Function to load and preprocess images
def load_and_preprocess_image(image_path_tensor, label):
    # Convert the tensor to a string
    image_path = tf.strings.join([image_save_folder, "/", image_path_tensor])

    # Read and preprocess the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0

    return img, label
# Define batch size
batch_size = 10
# Create TensorFlow datasets for training and testing
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_and_preprocess_image)
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_dataset = test_dataset.map(load_and_preprocess_image)
test_dataset = test_dataset.batch(batch_size)

# Create a new model with VGG16
pretrained_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in pretrained_model.layers:
    layer.trainable = False

model = Sequential()
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(102, activation='softmax'))

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"]
)

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

model.save("vgg16_102category_flower_dataset.h5")

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Save the plot as an image file (e.g., PNG)
plt.savefig('accuracy_plot.png')

# Show the plot
plt.show()

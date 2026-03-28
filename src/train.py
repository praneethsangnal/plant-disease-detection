from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Path using pathlib
BASE_DIR=Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/"data"
MODEL_DIR = BASE_DIR/"models"


# Settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    str(DATA_DIR),   # IMPORTANT: convert to string
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

val_data = datagen.flow_from_directory(
    str(DATA_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)

# Load pretrained model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# Custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
a=model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)
 
#Training and validation accuracy
print("\nFinal Training Accuracy:", a.history['accuracy'][-1])
print("Final Validation Accuracy:", a.history['val_accuracy'][-1]) 

# Plotting epochs vs training accuracy and epochs vs validation accuracy
plt.plot(range(1, 6), a.history['accuracy'], label='train_accuracy')
plt.plot(range(1, 6), a.history['val_accuracy'], label='val_accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.title("Model Training Progress")
plt.show()

# Save model
model.save(MODEL_DIR / "model.h5")

print("✅ Model trained and saved!")
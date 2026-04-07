from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import json

# ======================
# Paths
# ======================
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5


# ======================
# FIXED CLASS MAPPING
# ======================
class_names = sorted([folder.name for folder in DATA_DIR.iterdir() if folder.is_dir()])
class_indices = {name: i for i, name in enumerate(class_names)}

print("\nClass Mapping:")
print(class_indices)


# ======================
# Data Generators with basic data augmentation so model training becomes location independen/avoid overfitting
# ======================

datagen_basic = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

datagen_eff = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


def get_data(datagen):
    train = datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset="training",
        class_mode="categorical"
    )

    #Validation WITHOUT augmentation
    if datagen.preprocessing_function:
        val_datagen = ImageDataGenerator(
            preprocessing_function=datagen.preprocessing_function,
            validation_split=0.2
        )
    else:
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

    val = val_datagen.flow_from_directory(
        str(DATA_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset="validation",
        class_mode="categorical"
    )

    return train, val


# ======================
# Model Builders
# ======================

def build_cnn(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_mobilenet(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_efficientnet(num_classes):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# ======================
# Training Function
# ======================

def train_model(name, model_fn, datagen):
    print(f"\n Training {name}...\n")

    train_data, val_data = get_data(datagen)

    model = model_fn(len(class_names))  # use fixed class count

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    return model, history, final_train_acc, final_val_acc


# ======================
# Run All Models
# ======================

results = {}
histories = {}

models_config = {
    "cnn": (build_cnn, datagen_basic),
    "mobilenet": (build_mobilenet, datagen_basic),
    "efficientnet": (build_efficientnet, datagen_eff)
}

best_acc = 0
best_name = ""

for name, (model_fn, datagen) in models_config.items():
    model, history, train_acc, val_acc = train_model(name, model_fn, datagen)

    results[name] = {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc)
    }

    histories[name] = history.history

    if val_acc > best_acc:
        best_acc = val_acc
        best_name = name
        model.save(MODEL_DIR / "best_model.h5")

    tf.keras.backend.clear_session()
    del model


print(f"\nBest Model: {best_name} ({best_acc:.4f})")


# ======================
# Save Results
# ======================

results["class_indices"] = class_indices
results["best_model"] = best_name

with open(REPORT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=4)


# ======================
# Plot Accuracy
# ======================

plt.figure(figsize=(10,6))

for name, hist in histories.items():
    plt.plot(range(1,EPOCHS+1),hist['val_accuracy'], label=f"{name}_val")

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(REPORT_DIR / "accuracy_plot.png")
plt.show()

print("\nTraining complete!")
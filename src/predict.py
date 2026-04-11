from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
import cv2

# ======================
# Paths
# ======================
MODEL_PATH = Path("models/best_model.h5")
RESULTS_PATH = Path("reports/results.json")
IMAGE_PATH = Path("testimages/cornnorth.JPG")

IMG_SIZE = (224, 224)

# ======================
# Load model
# ======================
model = tf.keras.models.load_model(MODEL_PATH)

# ======================
# Load results
# ======================
with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

class_indices = results["class_indices"]
best_model_name = results["best_model"]

index_to_class = {v: k for k, v in class_indices.items()}

print("Best model:", best_model_name)


# ======================
# Grad-CAM
# ======================
# ======================
# Grad-CAM
# ======================
def get_gradcam_heatmap(img_array, model, pred_index=None):
    # 1. Reach into the Sequential model to get the EfficientNet base
    base_model = model.layers[0] 
    
    # 2. Get the last convolutional layer
    last_conv_layer = base_model.get_layer("Conv_1")

    # 3. FIX: Create a functional model using the BASE MODEL'S input
    # This bypasses the 'Sequential has no input' error.
    # We will get the features from the conv layer and the base model output.
    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )

    # 4. Use GradientTape
    with tf.GradientTape() as tape:
        # We run the img through the base_model part
        conv_outputs, base_preds = grad_model(img_array)
        
        # We run the base_preds through the rest of YOUR layers (GAP, Dense, etc.)
        # This manually reconstructs the 'Sequential' flow
        x = model.layers[1](base_preds) # GlobalAveragePooling2D
        x = model.layers[2](x)          # Dense 128
        full_predictions = model.layers[3](x) # Final Dense Layer (Softmax)

        if pred_index is None:
            pred_index = tf.argmax(full_predictions[0])
        
        class_channel = full_predictions[:, pred_index]

    # 5. Calculate gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Generate Heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

def overlay_heatmap(img, heatmap):
    img = np.array(img)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = (heatmap * 255).astype("uint8")

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    output_path = "gradcam_output.jpg"
    cv2.imwrite(output_path, superimposed_img)

    return output_path


# ======================
# Prediction
# ======================
def predict_image(img_input):

    # Handle both path and PIL image
    if isinstance(img_input, (str, Path)):
        img = Image.open(img_input).convert("RGB")
    else:
        img = img_input  # already PIL image

    img = img.resize(IMG_SIZE)

    img_array = image.img_to_array(img)

    # Correct preprocessing
    if best_model_name == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_array = preprocess_input(img_array)
    else:
        img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions))
    predicted_class = index_to_class[predicted_index]

    print("Predicted index:", predicted_index)
    print("Predicted class:", predicted_class)

    # Grad-CAM
    heatmap = get_gradcam_heatmap(img_array, model, pred_index=predicted_index)

    output_path = overlay_heatmap(img, heatmap)

    print(f"Grad-CAM saved at: {output_path}")

    return predicted_class, confidence,output_path


# ======================
# Run
# ======================
if __name__ == "__main__":

    label, conf, cam_path = predict_image(IMAGE_PATH)

    if "___" in label:
        plant, disease = label.split("___")
        print(f"\n Plant: {plant}")
        print(f" Disease: {disease}")
    else:
        print(f"\nPrediction: {label}")

    print(f"Confidence: {conf:.2f}")
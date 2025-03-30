import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image

# Path to the TFLite model
tflite_model_path = "models/child_seat_model.tflite"

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Get input shape and type from the model
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
img_size = input_shape[1]  # Assuming square input shape

print(f"Input shape: {input_shape}")
print(f"Input dtype: {input_dtype}")
print(f"Required image size: {img_size}x{img_size}")


# Function to preprocess images for TFLite model
def preprocess_image(image_path):
    """
    Preprocess image for the TFLite model
    """
    try:
        # Load image using TensorFlow IO operations
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)

        # Keep the original image for display
        display_img = tf.image.resize(img, [img_size, img_size])

        # Resize to match model's expected input
        img = tf.image.resize(img, [img_size, img_size])

        # Convert to numpy for the TFLite interpreter
        # TFLite models typically expect numpy arrays
        img_array = img.numpy()

        # Add batch dimension if needed
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)

        return img_array, display_img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None, None


# Function to process a batch of images
def process_images(directory, is_positive=True):
    """
    Process all images in a directory using the TFLite model and return predictions
    """
    confidences = []
    image_paths = []

    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return [], []

    for filename in os.listdir(directory)[:10]:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            image_paths.append(image_path)

            # Preprocess image
            input_data, display_img = preprocess_image(image_path)
            if input_data is None:
                continue

            # Set the input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run the inference
            interpreter.invoke()

            # Get the output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # IMPORTANT CHANGE: Flip the confidence interpretation
            # The TFLite model seems to be giving probability of "no child seat" as first value
            if output_data.shape[-1] == 2:
                confidence = output_data[0][0]  # Using first value
            else:
                confidence = 1.0 - output_data[0][0]  # Inverting single value

            confidences.append(float(confidence))

            category = "positive" if is_positive else "negative"
            print(f"Processed {category} image {filename}: confidence = {confidence:.4f}")

    return image_paths, confidences


# Test paths (adjust these to your actual test folders)
positive_images_path = "./cs3_dataset/test/child_seat"  # Directory with child seat images
negative_images_path = "./cs3_dataset/test/other"  # Directory with non-child seat images

# Process images
print("\nProcessing positive images...")
positive_image_paths, positive_confidences = process_images(positive_images_path, is_positive=True)
print(f"Processed {len(positive_image_paths)} positive images")

print("\nProcessing negative images...")
negative_image_paths, negative_confidences = process_images(negative_images_path, is_positive=False)
print(f"Processed {len(negative_image_paths)} negative images")

# Find best threshold
thresholds = np.arange(0.01, 1.0, 0.01)
best_threshold = 0.5  # Default
best_accuracy = 0.0

# Combine all confidences
all_positive_confidences = positive_confidences
all_negative_confidences = negative_confidences

# Try different thresholds to find the best one
for threshold in thresholds:
    # Count true positives and true negatives
    tp = sum(1 for conf in all_positive_confidences if conf >= threshold)
    tn = sum(1 for conf in all_negative_confidences if conf < threshold)

    # Calculate accuracy
    total = len(all_positive_confidences) + len(all_negative_confidences)
    accuracy = (tp + tn) / total if total > 0 else 0

    # Update best threshold if this is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.4f}")

# Calculate final metrics with the best threshold
tp = sum(1 for conf in all_positive_confidences if conf >= best_threshold)
fp = sum(1 for conf in all_negative_confidences if conf >= best_threshold)
fn = sum(1 for conf in all_positive_confidences if conf < best_threshold)
tn = sum(1 for conf in all_negative_confidences if conf < best_threshold)

precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print(f"Results at threshold {best_threshold:.2f}:")
print(f"  True Positives: {tp}")
print(f"  False Positives: {fp}")
print(f"  True Negatives: {tn}")
print(f"  False Negatives: {fn}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")


# Visualize a few images with predictions
def visualize_predictions(positive_paths, positive_confs, negative_paths, negative_confs, threshold, num_samples=3):
    """Visualize model predictions on sample images"""
    plt.figure(figsize=(12, 8))

    # Show positive samples
    for i in range(min(num_samples, len(positive_paths))):
        plt.subplot(2, num_samples, i + 1)
        img = tf.io.read_file(positive_paths[i])
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])

        # Normalize to [0,1] to avoid imshow warnings
        img_normalized = img / 255.0

        plt.imshow(img_normalized)

        confidence = positive_confs[i]
        color = 'green' if confidence >= threshold else 'red'
        plt.title(f"Positive: {confidence:.2f}", color=color)
        plt.axis("off")

    # Show negative samples
    for i in range(min(num_samples, len(negative_paths))):
        plt.subplot(2, num_samples, num_samples + i + 1)
        img = tf.io.read_file(negative_paths[i])
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])

        # Normalize to [0,1] to avoid imshow warnings
        img_normalized = img / 255.0

        plt.imshow(img_normalized)

        confidence = negative_confs[i]
        color = 'red' if confidence < threshold else 'green'
        plt.title(f"Negative: {confidence:.2f}", color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig('tflite_predictions.png')
    plt.show()


# Visualize some predictions
if len(positive_image_paths) > 0 and len(negative_image_paths) > 0:
    visualize_predictions(
        positive_image_paths[:3],
        positive_confidences[:3],
        negative_image_paths[:3],
        negative_confidences[:3],
        best_threshold
    )

import os
import time

import cv2
import numpy as np
import tensorflow as tf


class Webcam:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read_frame(self):
        """
        Read a frame from the webcam

        Returns:
            success: True if frame read successfully, False otherwise
            frame: the frame read (None if not successful)
        """
        if self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                return success, frame
        return False, None

    def release(self):
        """
        Release the video source when no longer needed
        """
        if self.cap.isOpened():
            self.cap.release()


class BoundingBox:
    def __init__(self, origin_x, origin_y, width, height):
        """
        Initialize a bounding box

        Args:
            origin_x: x-coordinate of top-left corner
            origin_y: y-coordinate of top-left corner
            width: width of bounding box
            height: height of bounding box
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height


class ChildSeatDetector:
    def __init__(self, model_path):
        """
        Initialize the child seat detector

        Args:
            model_path: Path to TFLite model file
        """
        # Detection state
        self.analyze_interval = 500  # milliseconds
        self.last_detection_time = 0
        self.child_seat_detected = False
        self.item_bounding_box = None
        self.last_item_detection_time = 0

        # Load TFLite model
        print(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            print(f"Input details: {self.input_details}")

            self.output_details = self.interpreter.get_output_details()
            print(f"Output details: {self.output_details}")

            # Get model input shape
            self.input_shape = self.input_details[0]['shape']
            print(f"Model input shape: {self.input_shape}")

            # Image size required by the model
            self.input_size = self.input_shape[1]  # Assuming square input
            print(f"Using input size: {self.input_size}x{self.input_size}")

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def preprocess_image(self, frame):
        """
        Preprocess image for the TFLite model

        Args:
            frame: RGB image frame

        Returns:
            Preprocessed numpy array ready for the model
        """
        try:
            # Convert to TensorFlow tensor
            img = tf.convert_to_tensor(frame)

            # Resize to match model's expected input
            img = tf.image.resize(img, [self.input_size, self.input_size])

            # Convert to numpy for the TFLite interpreter
            img_array = img.numpy()

            # Add batch dimension if needed
            if len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)

            return img_array
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None

    def detect(self, frame):
        """
        Analyze frame and detect child seats

        Args:
            frame: BGR image from webcam
        """
        current_time = time.time() * 1000

        # Skip if we haven't reached the analysis interval
        if current_time - self.last_detection_time < self.analyze_interval:
            return

        self.last_detection_time = current_time

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocess image
            input_data = self.preprocess_image(rgb_frame)

            if input_data is None:
                return

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

            # Run inference
            self.interpreter.invoke()

            # Get output data
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Reset detection state
            self.child_seat_detected = False
            self.item_bounding_box = None

            # Extract confidence value based on output shape
            # if output_data.shape[-1] == 2:
            #     confidence = output_data[0][1]  # Use second value (child seat probability)
            # else:
            #     confidence = output_data[0][0]  # Use single output value directly

            confidence = float(f"{output_data[0][0]:.4f}")

            print(f"Raw output: {output_data}")
            print(f"Confidence: {confidence:.4f}")

            # Determine if child seat is present based on the confidence
            if confidence > 0.99:  # We can adjust this threshold later if needed
                self.child_seat_detected = True
                print("CHILD SEAT DETECTED!")

                # Create bounding box for visualization
                h, w = frame.shape[:2]
                box_size = min(w, h) * 0.6
                center_x, center_y = w // 2, h // 2

                self.item_bounding_box = BoundingBox(
                    origin_x=int(center_x - box_size // 2),
                    origin_y=int(center_y - box_size // 2),
                    width=int(box_size),
                    height=int(box_size)
                )

                # Record detection time
                self.last_item_detection_time = time.time()

        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()

    def draw_item_bounding_box(self, frame):
        """
        Draw bounding box for detected item

        Args:
            frame: Frame to draw on

        Returns:
            Frame with bounding box drawn
        """
        if self.item_bounding_box:
            bb = self.item_bounding_box

            # Draw rectangle with yellow color instead of green
            cv2.rectangle(
                frame,
                (bb.origin_x, bb.origin_y),
                (bb.origin_x + bb.width, bb.origin_y + bb.height),
                (0, 255, 255),  # Yellow color (BGR)
                2  # Line thickness
            )

            # Add label
            label = ""
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Position label at top of bounding box
            label_origin_x = bb.origin_x
            label_origin_y = bb.origin_y - 10 if bb.origin_y > 20 else bb.origin_y + 20

            # Create semi-transparent background for text
            sub_img = frame[
                      label_origin_y - label_size[1]:label_origin_y,
                      label_origin_x:label_origin_x + label_size[0]
                      ]

            # Create a colored overlay
            overlay = sub_img.copy()
            overlay[:] = (255, 255, 0)  # Yellow background

            # Apply transparency
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, sub_img, 1 - alpha, 0, sub_img)

            # Put it back
            frame[
            label_origin_y - label_size[1]:label_origin_y,
            label_origin_x:label_origin_x + label_size[0]
            ] = sub_img

            # Draw text
            cv2.putText(
                frame,
                label,
                (label_origin_x, label_origin_y),
                font,
                font_scale,
                (255, 255, 255),  # Black color
                font_thickness
            )

        return frame


class Detector:
    def display_warning(self, frame):
        """
        Display warning message on frame

        Args:
            frame: Frame to display warning on

        Returns:
            Frame with warning message
        """
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)

        # Add text
        text = "WARNING"
        font_scale = 1.5
        thickness = 3
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (w - text_size[0]) // 2
        text_y = h // 2 - 50

        cv2.putText(
            overlay,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 255),  # Red color
            thickness
        )

        # Add explanation
        explanation = "Child seat detected!"
        font_scale_explanation = 0.8
        thickness_explanation = 2
        explanation_size, _ = cv2.getTextSize(explanation, font, font_scale_explanation, thickness_explanation)
        explanation_x = (w - explanation_size[0]) // 2
        explanation_y = text_y + 50

        cv2.putText(
            overlay,
            explanation,
            (explanation_x, explanation_y),
            font,
            font_scale_explanation,
            (255, 255, 255),  # White color
            thickness_explanation
        )

        # Blend with original frame
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1, 0)

        return frame

    def process_frame(self, frame, detector):
        """
        Process frame and apply detection

        Args:
            frame: Frame to process
            detector: ChildSeatDetector instance

        Returns:
            Processed frame
        """
        # Run detection
        detector.detect(frame)

        # Draw bounding box for detected item
        if detector.child_seat_detected:
            frame = detector.draw_item_bounding_box(frame)

            # Display warning for a period after detection
            time_since_detection = time.time() - detector.last_item_detection_time
            if time_since_detection < 5.0:  # Show warning for 5 seconds
                frame = self.display_warning(frame)

        return frame

    def run(self, model_path, video_source=0):
        """
        Run the detector on video feed

        Args:
            model_path: Path to TFLite model
            video_source: Video source index or file path
        """
        try:
            # Initialize webcam
            webcam = Webcam(video_source)

            # Initialize detector
            detector = ChildSeatDetector(model_path)

            # Create window
            window_name = "Child Seat Detector"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            print("Starting detection. Press 'q' to quit.")

            while True:
                # Read frame from webcam
                success, frame = webcam.read_frame()

                if not success:
                    print("Failed to read frame from webcam")
                    break

                # Process frame
                processed_frame = self.process_frame(frame, detector)

                # Display result
                cv2.imshow(window_name, processed_frame)

                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit command received")
                    break

        except Exception as e:
            print(f"Error in detector: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up
            if 'webcam' in locals():
                webcam.release()
            cv2.destroyAllWindows()
            print("Detector stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Child Seat Detector')
    parser.add_argument('--model', type=str, default='models/child_seat_model.tflite',
                        help='Path to TFLite model file')
    parser.add_argument('--source', type=int, default=0,
                        help='Video source (camera index, default: 0)')

    args = parser.parse_args()

    detector = Detector()
    detector.run(args.model, args.source)

import time

import cv2
import mediapipe as mp


class Webcam:
    """
    A class to handle webcam operations.

    Attributes:
        cap (cv2.VideoCapture): The video capture object for the webcam.
    """

    def __init__(self, index=0):
        """
        Initializes the webcam with the given index.

        Args:
            index (int): The index of the webcam to use. Default is 0.
        """
        self.cap = cv2.VideoCapture(index)

    def read_frame(self):
        """
        Reads a frame from the webcam.

        Returns:
            frame (numpy.ndarray): The captured frame, or None if the frame could not be read.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """
        Releases the webcam.
        """
        self.cap.release()


class HolisticDetector:
    """
    A class to handle holistic detection using MediaPipe.

    Attributes:
        holistic (mp.solutions.holistic.Holistic): The holistic model.
        drawing_spec (mp.solutions.drawing_utils.DrawingSpec): The drawing specifications for landmarks.
        connection_spec (mp.solutions.drawing_utils.DrawingSpec): The drawing specifications for connections.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initializes the holistic detector with the given confidence levels.

        Args:
            min_detection_confidence (float): Minimum confidence value for detection.
            min_tracking_confidence (float): Minimum confidence value for tracking.
        """
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=0)
        self.connection_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1)

    def process(self, rgb_frame):
        """
        Processes an RGB frame to detect holistic landmarks.

        Args:
            rgb_frame (numpy.ndarray): The RGB frame to process.

        Returns:
            results (mp.solutions.holistic.HolisticResults): The results of the holistic detection.
        """
        return self.holistic.process(rgb_frame)

    def draw_landmarks(self, frame, results):
        """
        Draws the detected landmarks on the frame.

        Args:
            frame (numpy.ndarray): The frame to draw on.
            results (mp.solutions.holistic.HolisticResults): The results of the holistic detection.
        """
        if results.face_landmarks:
            self._draw_landmarks(frame, results.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
        if results.left_hand_landmarks:
            self._draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self._draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    def _draw_landmarks(self, frame, landmarks, connections):
        """
        Draws the landmarks and connections on the frame.

        Args:
            frame (numpy.ndarray): The frame to draw on.
            landmarks (mp.framework.formats.landmark_pb2.NormalizedLandmarkList): The landmarks to draw.
            connections (list): The connections between landmarks.
        """
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            landmarks,
            connections,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.connection_spec
        )


class ObjectDetector:
    """
    A class to handle object detection using MediaPipe.

    Attributes:
        detect_delay (int): The delay between detections in milliseconds.
        last_detection_time (float): The time of the last detection.
        phone_detected (bool): Whether a phone was detected.
        phone_bounding_box (mp.tasks.components.containers.BoundingBox): The bounding box of the detected phone.
        last_phone_detection_time (float): The time of the last phone detection.
        detector (mp.tasks.vision.ObjectDetector): The object detector.
    """

    def __init__(self, model_path='./models/efficientdet_lite2_int8.tflite', detect_delay=200):
        """
        Initializes the object detector with the given model and detection delay.

        Args:
            model_path (str): The path to the model file.
            detect_delay (int): The delay between detections in milliseconds.
        """
        self.detect_delay = detect_delay
        self.last_detection_time = 0
        self.phone_detected = False
        self.phone_bounding_box = None
        self.last_phone_detection_time = 0

        with open(model_path, "rb") as f:
            model_data = f.read()

        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_data),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            max_results=2,
            result_callback=self.print_result
        )
        self.detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

    def print_result(self, result, output_image, timestamp_ms):
        """
        Callback function to process detection results.

        Args:
            result (mp.tasks.components.containers.DetectionResult): The detection results.
            output_image (mp.Image): The output image.
            timestamp_ms (int): The timestamp of the detection.
        """
        self.phone_detected = False
        self.phone_bounding_box = None
        for detection in result.detections:
            for category in detection.categories:
                if category.category_name == 'cell phone' and category.score > 0.5:
                    self.phone_detected = True
                    self.phone_bounding_box = detection.bounding_box
                    return

    def detect(self, rgb_frame):
        """
        Performs object detection on the given RGB frame.

        Args:
            rgb_frame (numpy.ndarray): The RGB frame to detect objects in.
        """
        current_time = time.time() * 1000
        if current_time - self.last_detection_time >= self.detect_delay:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self.detector.detect_async(mp_image, time.time_ns() // 1_000_000)
            self.last_detection_time = current_time

    def draw_phone_bounding_box(self, frame):
        """
        Draws the bounding box of the detected phone on the frame.

        Args:
            frame (numpy.ndarray): The frame to draw on.
        """
        if self.phone_detected and self.phone_bounding_box:
            frame_height, frame_width, _ = frame.shape
            x_min = int(self.phone_bounding_box.origin_x)
            y_min = int(self.phone_bounding_box.origin_y)
            x_max = int(self.phone_bounding_box.origin_x + self.phone_bounding_box.width)
            y_max = int(self.phone_bounding_box.origin_y + self.phone_bounding_box.height)
            color = (0, 0, 255)
            thickness = 1
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
            label = "Phone Detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            label_x = x_min
            label_y = max(0, y_min - 10)
            cv2.putText(frame, label, (label_x, label_y), font, font_scale, color, font_thickness)


class PhoneDetector:
    """
        A class to handle the phone detection process on a camera video stream.
    """

    @staticmethod
    def display_warning(frame, object_detector):
        """
        Displays a warning on the frame if a phone is detected.

        Args:
            frame (numpy.ndarray): The frame to display the warning on.
            object_detector (ObjectDetector): The object detector.
        """
        if time.time() - object_detector.last_phone_detection_time <= 3:
            cv2.putText(frame, "Person operating Phone", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if object_detector.phone_bounding_box:
                print(f"({time.time()}) <<< (!) WARNING >>> A Person operating a mobile phone detected! phone_bounding_box : {object_detector.phone_bounding_box}")
            else:
                print(f"({time.time()}) <<< (i) INFO >>> Person appears to have release the mobile phone, still displaying warning for a time period ...")

    def process_frame(self, webcam, holistic_detector, object_detector):
        """
        Processes a frame from the webcam, performing holistic and object detection.

        Args:
            webcam (Webcam): The webcam object.
            holistic_detector (HolisticDetector): The holistic detector.
            object_detector (ObjectDetector): The object detector.

        Returns:
            frame (numpy.ndarray): The processed frame, or None if the frame could not be read.
        """
        frame = webcam.read_frame()
        if frame is None:
            return None
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        holistic_results = holistic_detector.process(rgb_frame)
        object_detector.detect(rgb_frame)
        if object_detector.phone_detected and holistic_results.face_landmarks and (holistic_results.left_hand_landmarks or holistic_results.right_hand_landmarks):
            object_detector.last_phone_detection_time = time.time()
            self.display_warning(frame, object_detector)
        object_detector.draw_phone_bounding_box(frame)
        holistic_detector.draw_landmarks(frame, holistic_results)
        self.display_warning(frame, object_detector)
        return frame

    def run(self):
        """
        The main function to run the webcam stream and perform detection.
        """
        webcam = Webcam()
        holistic_detector = HolisticDetector()
        object_detector = ObjectDetector()
        while True:
            frame = self.process_frame(webcam, holistic_detector, object_detector)
            if frame is None:
                break
            cv2.imshow('Webcam Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    PhoneDetector().run()

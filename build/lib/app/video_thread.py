"""Video capture and processing thread for Qt application"""

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from core.face_analyser import get_face_analyser
from core.engine.face_swapper import swap_face
from core.engine.face_masking import (
    get_mouth_mask_without_poisson,
    get_mouth_mask_with_poisson,
    get_mask_center,
)


class VideoThread(QThread):
    """Worker thread for video capture and face swapping"""

    # Signals for thread-safe communication
    frame_ready = pyqtSignal(np.ndarray)
    fps_update = pyqtSignal(float)
    face_count_update = pyqtSignal(int)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.swap_enabled = False
        self.mouth_mask_enabled = False  # NEW: Mouth mask toggle
        self.source_face = None
        self.cap = None
        self.face_analyser = None

    def set_source_face(self, source_face):
        """Set the source face for swapping"""
        self.source_face = source_face

    def enable_swap(self, enabled):
        """Enable or disable face swapping"""
        self.swap_enabled = enabled

    def enable_mouth_mask(self, enabled):
        """Enable or disable mouth masking"""
        self.mouth_mask_enabled = enabled

    def run(self):
        """Main thread loop for video capture and processing"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.error_occurred.emit("Failed to open camera")
                return

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Initialize face analyser
            self.face_analyser = get_face_analyser()

            self.running = True
            fps_counter = FPSCounter()

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.error_occurred.emit("Failed to read frame from camera")
                    break

                # Flip frame horizontally for mirror effect
                # frame = cv2.flip(frame, 1)

                try:
                    # Detect faces in frame
                    faces = self.face_analyser.get(frame)
                    self.face_count_update.emit(len(faces))

                    # Apply face swapping if enabled and source face is set
                    if (
                        self.swap_enabled
                        and self.source_face is not None
                        and len(faces) > 0
                    ):
                        for face in faces:
                            # Apply mouth masking if enabled
                            if self.mouth_mask_enabled:
                                mask = get_mouth_mask_with_poisson(
                                    frame.shape, face.landmark_2d_106
                                )
                                swapped = swap_face(self.source_face, face, frame)

                                if mask is not None:
                                    center = get_mask_center(mask, frame.shape)

                                    frame = cv2.seamlessClone(
                                        src=frame,
                                        dst=swapped,
                                        mask=mask,
                                        p=center,
                                        flags=cv2.NORMAL_CLONE,
                                    )
                                else:
                                    frame = swapped
                            else:
                                # Regular face swap without masking
                                frame = swap_face(self.source_face, face, frame)
                    else:
                        # Draw bounding boxes around detected faces
                        for face in faces:
                            bbox = face.bbox.astype(int)
                            cv2.rectangle(
                                frame,
                                (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]),
                                (0, 255, 0),
                                2,
                            )

                except Exception as e:
                    # Continue processing even if face detection fails
                    pass

                # Emit processed frame
                self.frame_ready.emit(frame)

                # Update FPS
                fps = fps_counter.update()
                if fps > 0:
                    self.fps_update.emit(fps)

        except Exception as e:
            self.error_occurred.emit(f"Video thread error: {str(e)}")
        finally:
            self.cleanup()

    def stop(self):
        """Stop the video thread"""
        self.running = False
        self.wait()

    def cleanup(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()


class FPSCounter:
    """Simple FPS counter"""

    def __init__(self, avg_frames=30):
        self.avg_frames = avg_frames
        self.frame_times = []
        self.last_time = cv2.getTickCount()

    def update(self):
        """Update and return current FPS"""
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.last_time) / cv2.getTickFrequency()
        self.last_time = current_time

        self.frame_times.append(time_diff)
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)

        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0

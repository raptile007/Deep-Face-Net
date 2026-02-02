from PyQt6.QtCore import QThread, pyqtSignal
import cv2
from pathlib import Path
import time
from core.engine.face_swapper import swap_face
from core.face_analyser import get_face_analyser

class FileProcessingThread(QThread):
    """
    Background thread for processing video/image files.
    """
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished_processing = pyqtSignal(str)  # Emits output path
    error_occurred = pyqtSignal(str)

    def __init__(self, source_face, target_path, output_path=None):
        super().__init__()
        self.source_face = source_face
        self.target_path = Path(target_path)
        self.output_path = output_path
        self._is_running = True

    def run(self):
        try:
            if not self.target_path.exists():
                raise FileNotFoundError(f"Target file not found: {self.target_path}")

            # Determine if target is image or video
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            target_ext = self.target_path.suffix.lower()

            if not self.output_path:
                self.output_path = str(self.target_path.parent / f"{self.target_path.stem}_swapped{self.target_path.suffix}")

            self.status_update.emit("Initializing face analyser...")
            face_analyser = get_face_analyser()

            if target_ext in image_extensions:
                self.process_image(face_analyser)
            elif target_ext in video_extensions:
                self.process_video(face_analyser)
            else:
                raise ValueError(f"Unsupported file format: {target_ext}")

        except Exception as e:
            self.error_occurred.emit(str(e))

    def process_image(self, face_analyser):
        self.status_update.emit("Processing image...")
        img = cv2.imread(str(self.target_path))
        if img is None:
            raise ValueError("Failed to load target image")

        faces = face_analyser.get(img)
        if len(faces) == 0:
            self.status_update.emit("Warning: No faces found in target image")
            # We still save the original image or could error out.
            # Let's just save the original for now to be safe, or just stopping.
            # Usually better to process what we can.
        
        for face in faces:
            img = swap_face(self.source_face, face, img)

        cv2.imwrite(str(self.output_path), img)
        self.progress_update.emit(100)
        self.status_update.emit("Image processing complete")
        self.finished_processing.emit(str(self.output_path))

    def process_video(self, face_analyser):
        self.status_update.emit("Opening video...")
        cap = cv2.VideoCapture(str(self.target_path))
        if not cap.isOpened():
            raise ValueError("Failed to open target video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine output format - prefer mp4
        # If output_path doesn't end in mp4, we might have codec issues with mp4v
        # simple check/fix
        out_path_obj = Path(self.output_path)
        if out_path_obj.suffix.lower() != '.mp4':
             out_path_obj = out_path_obj.with_suffix('.mp4')
             self.output_path = str(out_path_obj)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            self.status_update.emit("Warning: 'mp4v' codec failed. Trying 'avc1'...")
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise ValueError(f"Failed to initialize video writer for {self.output_path}")

        print(f"Video Writer initialized: {self.output_path} ({width}x{height} @ {fps}fps)")

        frame_count = 0
        start_time = time.time()

        self.status_update.emit("Processing video frames...")
        
        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                break

            # Process Faces
            faces = face_analyser.get(frame)
            for face in faces:
                frame = swap_face(self.source_face, face, frame)

            out.write(frame)
            
            frame_count += 1
            if frame_count % 5 == 0: # Update progress every few frames to reduce signal overhead
                progress = int((frame_count / total_frames) * 100)
                self.progress_update.emit(progress)
                
                # Optional: Update status with more info occasionally
                if frame_count % 30 == 0:
                     elapsed = time.time() - start_time
                     fps_process = frame_count / elapsed if elapsed > 0 else 0
                     self.status_update.emit(f"Processing: {progress}% - {fps_process:.1f} FPS")

        cap.release()
        out.release()

        if self._is_running:
            self.progress_update.emit(100)
            self.status_update.emit("Video processing complete")
            self.finished_processing.emit(str(self.output_path))
        else:
            self.status_update.emit("Processing cancelled")

    def stop(self):
        self._is_running = False
        self.wait()

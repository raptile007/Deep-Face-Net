"""
Qt Deepfake Application - Real-time Face Swapping
A professional PyQt6 application for live deepfake face swapping
"""

import sys
import cv2
import numpy as np
import json
from pathlib import Path

# Try to import pyvirtualcam (optional dependency)
try:
    import pyvirtualcam
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False
    print("pyvirtualcam not available - virtual camera feature disabled")
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QStatusBar,
    QFrame,
    QCheckBox,
    QTabWidget,
    QProgressBar,
    QDialog,
    QLineEdit,
    QComboBox,
)
from PyQt6.QtCore import Qt, QTimer, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QDesktopServices

from app.video_thread import VideoThread
from app.drag_drop_widget import DragDropWidget
from app.file_processing_thread import FileProcessingThread
from core.face_analyser import get_face_analyser


class CameraDetectionThread(QThread):
    """Background thread for detecting available cameras"""
    
    cameras_detected = pyqtSignal(list)  # Emits list of camera indices
    
    def __init__(self, max_cameras=6):
        super().__init__()
        self.max_cameras = max_cameras
    
    def run(self):
        """Detect available cameras in background"""
        available = []
        # Test camera indices 0-5 (covers most cases, faster than 0-10)
        for i in range(self.max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify camera works
                # Set a short timeout to avoid long waits
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        
        self.cameras_detected.emit(available)


class SettingsDialog(QDialog):
    """Settings dialog for configuring app preferences"""
    
    def __init__(self, current_working_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.working_dir = current_working_dir
        
        layout = QVBoxLayout()
        
        # Working Directory Section
        dir_group = QGroupBox("Working Directory")
        dir_layout = QVBoxLayout()
        
        info_label = QLabel("All file browsers will start from this directory:")
        info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        dir_layout.addWidget(info_label)
        
        # Directory display and browse
        dir_row = QHBoxLayout()
        self.dir_input = QLineEdit(self.working_dir)
        self.dir_input.setReadOnly(True)
        dir_row.addWidget(self.dir_input)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_directory)
        dir_row.addWidget(browse_btn)
        
        dir_layout.addLayout(dir_row)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        save_btn.setStyleSheet("background-color: #4CAF50; min-width: 80px;")
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setStyleSheet("background-color: #555; min-width: 80px;")
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QGroupBox {
                font-weight: bold;
                color: #ffffff;
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel { color: #ffffff; }
            QLineEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
            }
        """)
    
    def browse_directory(self):
        """Open directory browser"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            self.working_dir
        )
        if directory:
            self.working_dir = directory
            self.dir_input.setText(directory)
    
    def get_working_dir(self):
        """Return the selected working directory"""
        return self.working_dir


class DeepfakeApp(QMainWindow):
    """Main application window for deepfake face swapping"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Face Net - Advanced Face Swapping")
        self.setGeometry(100, 100, 1200, 800)

        # Application state
        self.source_image = None
        self.source_face = None
        self.video_thread = None
        self.is_capturing = False
        self.selected_camera_index = 0  # Default camera
        self.available_cameras = []
        self.virtual_cam = None  # Virtual camera instance
        self.virtual_cam_enabled = False  # Virtual camera state
        
        # Offline processing state
        self.offline_source_image = None
        self.offline_source_face = None
        self.target_file_path = None
        self.processing_thread = None
        self.last_output_path = None
        
        # Settings
        self.settings_file = Path.home() / ".deepfacenet_settings.json"
        self.working_dir = self.load_settings()

        # Initialize UI
        self.init_ui()

        # Apply modern styling
        self.apply_styles()

    def init_ui(self):
        """Initialize the user interface"""
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create Tab Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Live Camera
        self.live_tab = QWidget()
        self.setup_live_tab()
        self.tabs.addTab(self.live_tab, "Live Camera")

        # Tab 2: Offline Processing
        self.offline_tab = QWidget()
        self.setup_offline_tab()
        self.tabs.addTab(self.offline_tab, "Offline Processing")

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.fps_label = QLabel("FPS: 0")
        self.face_count_label = QLabel("Faces: 0")
        self.status_bar.addWidget(self.status_label, stretch=1)
        self.status_bar.addPermanentWidget(self.face_count_label)
        self.status_bar.addPermanentWidget(self.fps_label)
        
        # Add Settings button to status bar
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self.open_settings)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                padding: 5px 15px;
                margin-left: 10px;
            }
            QPushButton:hover { background-color: #666; }
        """)
        self.status_bar.addPermanentWidget(settings_btn)
        
        # Detect and populate available cameras
        self.detect_and_populate_cameras()

    def setup_live_tab(self):
        """Setup the Live Camera tab"""
        layout = QHBoxLayout(self.live_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # Left panel - Video display
        left_panel = self.create_video_panel()
        layout.addWidget(left_panel, stretch=3)

        # Right panel - Controls
        right_panel = self.create_control_panel()
        layout.addWidget(right_panel, stretch=1)

    def setup_offline_tab(self):
        """Setup the Offline Processing tab"""
        layout = QHBoxLayout(self.offline_tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Left Column: Inputs
        input_layout = QVBoxLayout()
        
        # 1. Source Image Selection
        source_group = QGroupBox("1. Select Source Face")
        source_inner = QVBoxLayout()
        
        # Header with Clear Button
        source_header = QHBoxLayout()
        source_header.addStretch()
        self.btn_clear_source = QPushButton("✕")
        self.btn_clear_source.setFixedSize(24, 24)
        self.btn_clear_source.setToolTip("Clear source selection")
        self.btn_clear_source.setEnabled(False)
        self.btn_clear_source.clicked.connect(self.clear_offline_source)
        self.btn_clear_source.setStyleSheet("""
            QPushButton {
                background-color: #444; color: #fff; border-radius: 12px; font-weight: bold; padding: 0;
            }
            QPushButton:hover { background-color: #f44336; }
            QPushButton:disabled { background-color: #333; color: #555; }
        """)
        source_header.addWidget(self.btn_clear_source)
        source_inner.addLayout(source_header)
        
        self.offline_source_preview = QLabel()
        self.offline_source_preview.setFixedSize(200, 200)
        self.offline_source_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offline_source_preview.setStyleSheet("QLabel { background-color: #2a2a2a; border: 2px solid #444; border-radius: 8px; }")
        self.offline_source_preview.setText("No source selected")
        
        btn_select_source = QPushButton("Browse Source Image...")
        btn_select_source.clicked.connect(self.select_offline_source_image)
        
        source_inner.addWidget(self.offline_source_preview, alignment=Qt.AlignmentFlag.AlignCenter)
        source_inner.addWidget(btn_select_source)
        source_group.setLayout(source_inner)
        input_layout.addWidget(source_group)

        # 2. Target File Selection
        target_group = QGroupBox("2. Select Target Video/Image")
        target_inner = QVBoxLayout()
        
        # Header with Clear Button
        target_header = QHBoxLayout()
        target_header.addStretch()
        self.btn_clear_target = QPushButton("✕")
        self.btn_clear_target.setFixedSize(24, 24)
        self.btn_clear_target.setToolTip("Clear target selection")
        self.btn_clear_target.setEnabled(False)
        self.btn_clear_target.clicked.connect(self.clear_target_file)
        self.btn_clear_target.setStyleSheet("""
            QPushButton {
                background-color: #444; color: #fff; border-radius: 12px; font-weight: bold; padding: 0;
            }
            QPushButton:hover { background-color: #f44336; }
            QPushButton:disabled { background-color: #333; color: #555; }
        """)
        target_header.addWidget(self.btn_clear_target)
        target_inner.addLayout(target_header)
        
        self.drop_widget = DragDropWidget()
        self.drop_widget.setMinimumHeight(150)
        self.drop_widget.file_dropped.connect(self.handle_file_drop)
        
        btn_browse_target = QPushButton("Browse Target File...")
        btn_browse_target.clicked.connect(self.browse_target_file)
        
        target_inner.addWidget(self.drop_widget)
        target_inner.addWidget(btn_browse_target)
        target_group.setLayout(target_inner)
        input_layout.addWidget(target_group)

        layout.addLayout(input_layout, stretch=1)

        # Right Column: Processing & Status
        process_layout = QVBoxLayout()
        
        # Info / Status
        status_group = QGroupBox("3. Process")
        status_inner = QVBoxLayout()
        status_inner.setSpacing(15)

        self.process_info_label = QLabel("Select a source face and a target file to begin.")
        self.process_info_label.setWordWrap(True)
        self.process_info_label.setStyleSheet("color: #aaa; font-style: italic;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
        """)

        self.btn_start_process = QPushButton("Start Processing")
        self.btn_start_process.setEnabled(False)
        self.btn_start_process.setMinimumHeight(50)
        self.btn_start_process.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """)
        self.btn_start_process.clicked.connect(self.start_offline_processing)

        status_inner.addWidget(self.process_info_label)
        status_inner.addWidget(self.progress_bar)
        status_inner.addWidget(self.btn_start_process)

        # Result Actions
        actions_layout = QHBoxLayout()
        
        self.btn_open_file = QPushButton("Open File")
        self.btn_open_file.setEnabled(False)
        self.btn_open_file.clicked.connect(self.open_output_file)
        self.btn_open_file.setStyleSheet("background-color: #555;")
        
        self.btn_open_folder = QPushButton("Show in Folder")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setStyleSheet("background-color: #555;")
        
        actions_layout.addWidget(self.btn_open_file)
        actions_layout.addWidget(self.btn_open_folder)
        
        status_inner.addLayout(actions_layout)

        # Result Preview (Hidden by default)
        self.result_preview_label = QLabel()
        self.result_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview_label.setVisible(False)
        self.result_preview_label.setStyleSheet("border: 2px solid #4caf50; border-radius: 5px;")
        status_inner.addWidget(self.result_preview_label)

        status_inner.addStretch()
        
        status_group.setLayout(status_inner)
        process_layout.addWidget(status_group)
        
        layout.addLayout(process_layout, stretch=1)

    # --- Live Tab Methods ---

    def create_video_panel(self):
        """Create the video display panel"""
        group = QGroupBox("Live Video Feed")
        layout = QVBoxLayout()

        # Minimal clear button in top-right
        header = QHBoxLayout()
        header.addStretch()
        
        # Clear button
        self.clear_feed_btn = QPushButton("✕")
        self.clear_feed_btn.setFixedSize(20, 20)
        self.clear_feed_btn.setToolTip("Clear feed")
        self.clear_feed_btn.clicked.connect(self.clear_video_feed)
        self.clear_feed_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: #fff;
                border-radius: 10px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover { background-color: #666; }
        """)
        header.addWidget(self.clear_feed_btn)
        layout.addLayout(header)

        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; border: 2px solid #333; border-radius: 8px; }"
        )
        self.video_label.setText("Camera feed will appear here")
        layout.addWidget(self.video_label)

        group.setLayout(layout)
        return group

    def create_control_panel(self):
        """Create the control panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # Source image section
        source_group = QGroupBox("Source Image")
        source_layout = QVBoxLayout()

        # Header with Clear Button
        source_header = QHBoxLayout()
        source_header.addStretch()
        self.btn_clear_live_source = QPushButton("✕")
        self.btn_clear_live_source.setFixedSize(24, 24)
        self.btn_clear_live_source.setToolTip("Clear source selection")
        self.btn_clear_live_source.setEnabled(False)
        self.btn_clear_live_source.clicked.connect(self.clear_live_source)
        self.btn_clear_live_source.setStyleSheet("""
            QPushButton {
                background-color: #444; color: #fff; border-radius: 12px; font-weight: bold; padding: 0;
            }
            QPushButton:hover { background-color: #f44336; }
            QPushButton:disabled { background-color: #333; color: #555; }
        """)
        source_header.addWidget(self.btn_clear_live_source)
        source_layout.addLayout(source_header)

        self.source_preview = QLabel()
        self.source_preview.setFixedSize(200, 200)
        self.source_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_preview.setStyleSheet(
            "QLabel { background-color: #2a2a2a; border: 2px solid #444; border-radius: 8px; }"
        )
        self.source_preview.setText("No image selected")
        source_layout.addWidget(
            self.source_preview, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.select_source_btn = QPushButton("Select Source Image")
        self.select_source_btn.clicked.connect(self.select_source_image)
        source_layout.addWidget(self.select_source_btn)

        self.source_status = QLabel("No source loaded")
        self.source_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_status.setStyleSheet("color: #888;")
        source_layout.addWidget(self.source_status)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Camera controls section
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout()

        # Camera selection
        camera_select_label = QLabel("Select Camera:")
        camera_select_label.setStyleSheet("color: #ffffff; font-weight: bold; margin-top: 5px;")
        camera_layout.addWidget(camera_select_label)
        
        camera_select_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumHeight(30)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selection_changed)
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 2px solid #444;
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox:hover {
                border-color: #4CAF50;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #ffffff;
                selection-background-color: #4CAF50;
            }
        """)
        camera_select_row.addWidget(self.camera_combo)
        
        self.refresh_cameras_btn = QPushButton("🔄")
        self.refresh_cameras_btn.setFixedSize(30, 30)
        self.refresh_cameras_btn.setToolTip("Refresh camera list")
        self.refresh_cameras_btn.clicked.connect(self.refresh_cameras)
        self.refresh_cameras_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        camera_select_row.addWidget(self.refresh_cameras_btn)
        camera_layout.addLayout(camera_select_row)

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.toggle_camera)
        self.start_btn.setMinimumHeight(40)
        camera_layout.addWidget(self.start_btn)

        self.swap_btn = QPushButton("Enable Face Swap")
        self.swap_btn.clicked.connect(self.toggle_swap)
        self.swap_btn.setEnabled(False)
        self.swap_btn.setMinimumHeight(40)
        camera_layout.addWidget(self.swap_btn)

        # Mouth mask checkbox
        self.mouth_mask_checkbox = QCheckBox("Enable Mouth Mask")
        self.mouth_mask_checkbox.setEnabled(False)
        self.mouth_mask_checkbox.stateChanged.connect(self.toggle_mouth_mask)
        self.mouth_mask_checkbox.setStyleSheet(
            "QCheckBox { color: #ffffff; padding: 5px; }"
            "QCheckBox::indicator { width: 18px; height: 18px; }"
        )
        camera_layout.addWidget(self.mouth_mask_checkbox)

        # Virtual camera checkbox (only if available)
        if VIRTUAL_CAM_AVAILABLE:
            self.virtual_cam_checkbox = QCheckBox("Enable Virtual Camera")
            self.virtual_cam_checkbox.setEnabled(False)
            self.virtual_cam_checkbox.stateChanged.connect(self.toggle_virtual_camera)
            self.virtual_cam_checkbox.setStyleSheet(
                "QCheckBox { color: #ffffff; padding: 5px; }"
                "QCheckBox::indicator { width: 18px; height: 18px; }"
            )
            self.virtual_cam_checkbox.setToolTip("Stream to virtual webcam for OBS/Zoom/Discord")
            camera_layout.addWidget(self.virtual_cam_checkbox)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Info section
        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "1. Select a source image\n"
            "2. Start the camera\n"
            "3. Enable face swap\n\n"
            "The source face will be\n"
            "swapped onto detected faces\n"
            "in the live video feed."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #aaa; font-size: 11px;")
        info_layout.addWidget(info_text)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def detect_and_populate_cameras(self):
        """Detect cameras in background and populate the combo box"""
        self.status_label.setText("Detecting cameras...")
        self.camera_combo.setEnabled(False)
        self.refresh_cameras_btn.setEnabled(False)
        
        # Start background detection thread
        self.camera_detection_thread = CameraDetectionThread(max_cameras=6)
        self.camera_detection_thread.cameras_detected.connect(self.on_cameras_detected)
        self.camera_detection_thread.start()

    def on_cameras_detected(self, available_cameras):
        """Handle camera detection results from background thread"""
        self.available_cameras = available_cameras
        
        self.camera_combo.clear()
        if self.available_cameras:
            for cam_idx in self.available_cameras:
                self.camera_combo.addItem(f"Camera {cam_idx}", cam_idx)
            self.status_label.setText(f"Found {len(self.available_cameras)} camera(s)")
            # Set default selection
            if self.selected_camera_index in self.available_cameras:
                idx = self.available_cameras.index(self.selected_camera_index)
                self.camera_combo.setCurrentIndex(idx)
        else:
            self.camera_combo.addItem("No cameras found", -1)
            self.selected_camera_index = -1
            self.status_label.setText("No cameras detected")
        
        self.camera_combo.setEnabled(True)
        self.refresh_cameras_btn.setEnabled(True)

    def refresh_cameras(self):
        """Refresh the camera list"""
        if self.is_capturing:
            QMessageBox.warning(
                self,
                "Camera Active",
                "Please stop the camera before refreshing the camera list."
            )
            return
        self.detect_and_populate_cameras()

    def on_camera_selection_changed(self, index):
        """Handle camera selection change"""
        if index >= 0:
            self.selected_camera_index = self.camera_combo.itemData(index)
            if self.selected_camera_index >= 0:
                self.status_label.setText(f"Selected Camera {self.selected_camera_index}")

    def select_source_image(self):
        """Open file dialog to select source image for live mode"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source Image",
            self.working_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )

        if file_path:
            try:
                # Show loading indicator
                self.status_label.setText("Loading source image...")
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                QApplication.processEvents()  # Keep UI responsive
                
                # Load image
                self.source_image = cv2.imread(file_path)
                if self.source_image is None:
                    raise ValueError("Failed to load image")

                # Extract face from source image
                self.status_label.setText("Analyzing face...")
                QApplication.processEvents()
                face_analyser = get_face_analyser()
                faces = face_analyser.get(self.source_image)

                if len(faces) == 0:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(
                        self,
                        "No Face Detected",
                        "No face was detected in the selected image. Please choose an image with a clear face.",
                    )
                    self.status_label.setText("Ready")
                    return

                # Use the first detected face
                self.source_face = faces[0]

                # Display preview
                self.display_source_preview(self.source_image, self.source_preview)
                self.source_status.setText(f"✓ Face detected")
                self.source_status.setStyleSheet("color: #4CAF50;")
                self.btn_clear_live_source.setEnabled(True)

                # Update video thread if running
                if self.video_thread is not None:
                    self.video_thread.set_source_face(self.source_face)
                    self.swap_btn.setEnabled(True)

                self.status_label.setText("Source image loaded successfully")
                QApplication.restoreOverrideCursor()

            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(
                    self, "Error", f"Failed to process image: {str(e)}"
                )
                self.status_label.setText("Ready")

    def clear_live_source(self):
        """Clear the live source image selection"""
        self.source_image = None
        self.source_face = None
        self.source_preview.clear()
        self.source_preview.setText("No image selected")
        self.source_status.setText("No source loaded")
        self.source_status.setStyleSheet("color: #888;")
        self.btn_clear_live_source.setEnabled(False)
        
        # Update video thread if running
        if self.video_thread is not None:
            self.video_thread.set_source_face(None)
            self.swap_btn.setEnabled(False)
            # If swap was enabled, disable it
            if self.video_thread.swap_enabled:
                self.toggle_swap()
        
        self.status_label.setText("Source image cleared")

    def display_source_preview(self, image, label_widget):
        """Display source image preview on a specific label"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to fit preview
        h, w = rgb_image.shape[:2]
        aspect = w / h
        if aspect > 1:
            new_w = 200
            new_h = int(200 / aspect)
        else:
            new_h = 200
            new_w = int(200 * aspect)

        resized = cv2.resize(rgb_image, (new_w, new_h))

        # Convert to QPixmap
        q_image = QImage(
            resized.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)

        label_widget.setPixmap(pixmap)

    def toggle_camera(self):
        """Start or stop camera capture"""
        if not self.is_capturing:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start video capture"""
        try:
            # Validate camera selection
            if self.selected_camera_index < 0:
                QMessageBox.warning(
                    self,
                    "No Camera",
                    "No camera selected. Please select a valid camera from the dropdown."
                )
                return
            
            # Create and start video thread with selected camera
            self.video_thread = VideoThread(camera_index=self.selected_camera_index)
            self.video_thread.frame_ready.connect(self.update_frame)
            self.video_thread.fps_update.connect(self.update_fps)
            self.video_thread.face_count_update.connect(self.update_face_count)
            self.video_thread.error_occurred.connect(self.handle_error)

            # Set source face if available
            if self.source_face is not None:
                self.video_thread.set_source_face(self.source_face)
                self.swap_btn.setEnabled(True)

            self.video_thread.start()

            self.is_capturing = True
            self.start_btn.setText("Stop Camera")
            self.start_btn.setStyleSheet("background-color: #f44336;")
            self.camera_combo.setEnabled(False)  # Disable camera selection while running
            self.refresh_cameras_btn.setEnabled(False)
            
            # Enable virtual camera checkbox if available
            if VIRTUAL_CAM_AVAILABLE:
                self.virtual_cam_checkbox.setEnabled(True)
            
            self.status_label.setText(f"Camera {self.selected_camera_index} started")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start camera: {str(e)}")

    def stop_camera(self):
        """Stop video capture"""
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None

        self.is_capturing = False
        self.start_btn.setText("Start Camera")
        self.start_btn.setStyleSheet("")
        self.camera_combo.setEnabled(True)  # Re-enable camera selection
        self.refresh_cameras_btn.setEnabled(True)
        self.swap_btn.setEnabled(False)
        self.swap_btn.setText("Enable Face Swap")
        self.swap_btn.setStyleSheet("")
        self.mouth_mask_checkbox.setEnabled(False)
        self.mouth_mask_checkbox.setChecked(False)
        
        # Stop and disable virtual camera
        if VIRTUAL_CAM_AVAILABLE:
            if self.virtual_cam_enabled:
                self.virtual_cam_checkbox.setChecked(False)
            self.virtual_cam_checkbox.setEnabled(False)
        
        # Automatically clear the video feed
        self.clear_video_feed()
        
        self.status_label.setText("Camera stopped")
        self.fps_label.setText("FPS: 0")
        self.face_count_label.setText("Faces: 0")

    def clear_video_feed(self):
        """Clear the video feed display"""
        # Remove any pixmap and force text display
        self.video_label.setPixmap(QPixmap())
        self.video_label.clear()
        self.video_label.setText("Camera feed will appear here")
        self.video_label.repaint()  # Force immediate repaint

    def toggle_swap(self):
        """Toggle face swapping on/off"""
        if self.video_thread is not None:
            current_state = self.video_thread.swap_enabled
            new_state = not current_state
            self.video_thread.enable_swap(new_state)

            if new_state:
                self.swap_btn.setText("Disable Face Swap")
                self.swap_btn.setStyleSheet("background-color: #FF9800;")
                self.status_label.setText("Face swapping enabled")
                self.mouth_mask_checkbox.setEnabled(True)
            else:
                self.swap_btn.setText("Enable Face Swap")
                self.swap_btn.setStyleSheet("")
                self.status_label.setText("Face swapping disabled")
                self.mouth_mask_checkbox.setEnabled(False)
                self.mouth_mask_checkbox.setChecked(False)

    def toggle_mouth_mask(self, state):
        """Toggle mouth masking on/off"""
        if self.video_thread is not None:
            enabled = state == Qt.CheckState.Checked.value
            self.video_thread.enable_mouth_mask(enabled)
            
            if enabled:
                self.status_label.setText("Mouth masking enabled")
            else:
                self.status_label.setText("Mouth masking disabled")

    def toggle_virtual_camera(self, state):
        """Toggle virtual camera output"""
        if not VIRTUAL_CAM_AVAILABLE:
            return
            
        enabled = state == Qt.CheckState.Checked.value
        
        if enabled:
            # Check if camera is running
            if not self.is_capturing or self.video_thread is None:
                self.virtual_cam_checkbox.setChecked(False)
                QMessageBox.warning(
                    self,
                    "Camera Not Running",
                    "Please start the camera before enabling virtual camera."
                )
                return
            
            try:
                # Get actual camera resolution from video thread
                if hasattr(self.video_thread, 'cap') and self.video_thread.cap is not None:
                    width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    # Fallback to common resolution
                    width, height = 640, 480
                
                # Create virtual camera with actual camera resolution
                self.virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
                self.virtual_cam_enabled = True
                self.status_label.setText(f"Virtual camera started: {self.virtual_cam.device} ({width}x{height})")
            except Exception as e:
                self.virtual_cam_enabled = False
                self.virtual_cam_checkbox.setChecked(False)
                QMessageBox.critical(
                    self,
                    "Virtual Camera Error",
                    f"Failed to start virtual camera:\n{str(e)}\n\nOn Linux, install v4l2loopback:\nsudo modprobe v4l2loopback"
                )
        else:
            if self.virtual_cam is not None:
                self.virtual_cam.close()
                self.virtual_cam = None
            self.virtual_cam_enabled = False
            self.status_label.setText("Virtual camera stopped")

    def update_frame(self, frame):
        """Update video display with new frame"""
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.video_label.setPixmap(scaled_pixmap)
        
        # Send to virtual camera if enabled
        if self.virtual_cam_enabled and self.virtual_cam is not None:
            try:
                # Send frame at original resolution (matches virtual camera size)
                self.virtual_cam.send(frame)
            except Exception as e:
                print(f"Virtual camera error: {e}")
                self.virtual_cam_enabled = False
                if VIRTUAL_CAM_AVAILABLE:
                    self.virtual_cam_checkbox.setChecked(False)

    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_face_count(self, count):
        """Update face count display"""
        self.face_count_label.setText(f"Faces: {count}")

    def handle_error(self, error_msg):
        """Handle errors from video thread"""
        QMessageBox.critical(self, "Error", error_msg)
        self.stop_camera()

    # --- Offline Tab Methods ---

    def select_offline_source_image(self):
        """Select source image for offline mode"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source Image",
            self.working_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )

        if file_path:
            try:
                # Show loading indicator
                self.process_info_label.setText("Loading source image...")
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                QApplication.processEvents()
                
                # Load image
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Failed to load image")

                # Extract face
                self.process_info_label.setText("Analyzing face...")
                QApplication.processEvents()
                face_analyser = get_face_analyser()
                faces = face_analyser.get(image)

                if len(faces) == 0:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(self, "No Face", "No face detected in selected image.")
                    self.process_info_label.setText("Select a source face and a target file to begin.")
                    return

                self.offline_source_face = faces[0]
                self.offline_source_image = image
                
                # Display
                self.display_source_preview(image, self.offline_source_preview)
                self.btn_clear_source.setEnabled(True)
                self.check_offline_readiness()
                QApplication.restoreOverrideCursor()
                
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self, "Error", str(e))
                self.process_info_label.setText("Select a source face and a target file to begin.")

    def clear_offline_source(self):
        """Clear the offline source selection"""
        self.offline_source_image = None
        self.offline_source_face = None
        self.offline_source_preview.clear()
        self.offline_source_preview.setText("No source selected")
        self.btn_clear_source.setEnabled(False)
        self.check_offline_readiness()

    def handle_file_drop(self, file_path):
        """Handle dropped file for offline processing"""
        self.target_file_path = file_path
        # Update target directory memory when file is dropped
        self.last_target_dir = str(Path(file_path).parent)
        self.btn_clear_target.setEnabled(True)
        self.check_offline_readiness()
        self.process_info_label.setText(f"Target selected: {Path(file_path).name}")

    def browse_target_file(self):
        """Browse for target video/image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Target File",
            self.working_dir,
            "Video/Image Files (*.mp4 *.avi *.mov *.jpg *.png)",
        )
        if file_path:
            self.last_target_dir = str(Path(file_path).parent)
            self.handle_file_drop(file_path)
            # Update drag widget text manually if browsed
            self.drop_widget.setText(f"Selected: {Path(file_path).name}")
            self.drop_widget.setStyleSheet("""
                QLabel {
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    background-color: #252525;
                    color: #4CAF50;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 20px;
                }
            """)
            self.btn_clear_target.setEnabled(True)

    def clear_target_file(self):
        """Clear the target file selection"""
        self.target_file_path = None
        self.drop_widget.setText("Drag & Drop Video/Image Here")
        self.drop_widget.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 10px;
                background-color: #252525;
                color: #aaa;
                font-size: 14px;
                padding: 20px;
            }
            QLabel:hover {
                border-color: #4CAF50;
                background-color: #2d2d2d;
                color: #fff;
            }
        """)
        self.btn_clear_target.setEnabled(False)
        self.check_offline_readiness()
        self.process_info_label.setText("Select a source face and a target file to begin.")

    def check_offline_readiness(self):
        """Enable start button if all inputs are ready"""
        if self.offline_source_face is not None and self.target_file_path is not None:
            self.btn_start_process.setEnabled(True)
            self.process_info_label.setText("Ready to process.")
        else:
            self.btn_start_process.setEnabled(False)

    def start_offline_processing(self):
        """Start the offline processing thread"""
        if not self.target_file_path or self.offline_source_face is None:
            return

        self.btn_start_process.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.processing_thread = FileProcessingThread(
            self.offline_source_face, 
            self.target_file_path
        )
        self.processing_thread.progress_update.connect(self.progress_bar.setValue)
        self.processing_thread.status_update.connect(self.process_info_label.setText)
        self.processing_thread.finished_processing.connect(self.on_processing_finished)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        
        self.processing_thread.start()

    def on_processing_finished(self, output_path):
        """Handle processing completion"""
        self.btn_start_process.setEnabled(True)
        self.last_output_path = output_path
        
        self.process_info_label.setText(f"Done! Saved to: {Path(output_path).name}")
        
        # Enable actions
        self.btn_open_file.setEnabled(True)
        self.btn_open_folder.setEnabled(True)
        self.btn_open_file.setStyleSheet("background-color: #2196F3;")
        self.btn_open_folder.setStyleSheet("background-color: #FF9800;")
        
        # Show preview if image
        if Path(output_path).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            self.show_result_preview(output_path)

    def show_result_preview(self, path):
        """Show preview of processed image"""
        try:
            image = cv2.imread(path)
            if image is not None:
                # Resize for preview (max 300px height)
                h, w = image.shape[:2]
                scale = min(1.0, 300/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized = cv2.resize(image, (new_w, new_h))
                
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
                self.result_preview_label.setPixmap(QPixmap.fromImage(q_image))
                self.result_preview_label.setVisible(True)
        except Exception as e:
            print(f"Preview error: {e}")

    def open_output_file(self):
        """Open the processed file"""
        if self.last_output_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output_path))

    def open_output_folder(self):
        """Open the folder containing the processed file"""
        if self.last_output_path:
            folder = str(Path(self.last_output_path).parent)
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def on_processing_error(self, error_msg):
        """Handle processing error"""
        self.btn_start_process.setEnabled(True)
        self.process_info_label.setText("Error occurred.")
        QMessageBox.critical(self, "Error", f"Processing failed: {error_msg}")

    def apply_styles(self):
        """Apply modern styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #aaa;
                padding: 10px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3a3a3a;
                color: #fff;
                font-weight: bold;
                border-bottom: 2px solid #4CAF50;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #ffffff;
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QLabel {
                color: #ffffff;
            }
            QStatusBar {
                background-color: #2a2a2a;
                color: #ffffff;
            }
            QStatusBar QLabel {
                color: #ffffff;
                padding: 2px 8px;
            }
            QCheckBox {
                color: white;
            }
        """)

    def load_settings(self):
        """Load settings from JSON file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    return settings.get('working_dir', str(Path.home()))
        except Exception as e:
            print(f"Error loading settings: {e}")
        return str(Path.home())
    
    def save_settings(self):
        """Save settings to JSON file"""
        try:
            settings = {
                'working_dir': self.working_dir
            }
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self.working_dir, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.working_dir = dialog.get_working_dir()
            self.save_settings()
            self.status_label.setText(f"Working directory updated: {Path(self.working_dir).name}")

    def closeEvent(self, event):
        """Handle application close event"""
        if self.is_capturing:
            self.stop_camera()
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = DeepfakeApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# Deep Face Net

<div align="center">

![Deep Face Net](https://img.shields.io/badge/Deep%20Face%20Net-Advanced%20Face%20Swapping-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![PyQt6](https://img.shields.io/badge/PyQt6-GUI-orange?style=for-the-badge&logo=qt)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv)

**A professional real-time face swapping application with both GUI and CLI support**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Screenshots](#screenshots) • [License](#license)

</div>

<img width="1189" height="932" alt="image" src="https://github.com/user-attachments/assets/ece0cdfb-7b12-4f03-a3e2-ed6bf06ede9c" />
<img width="1195" height="932" alt="image" src="https://github.com/user-attachments/assets/11e05b05-0581-4060-8642-672acc15e357" />

---

## Features

### Dual Interface
- **Modern GUI Application** - Professional PyQt6 interface with dark theme
- **Command Line Interface** - Powerful CLI for batch processing and automation

### Live Processing
- **Real-time Webcam Face Swapping** - Swap faces in live video feed
- **Multiple Camera Support** - Detect and switch between available cameras
- **Virtual Camera Output** - Stream to OBS, Zoom, Discord, and other apps (optional)
- **Mouth Mask Feature** - Preserve natural mouth movements

### Offline Processing
- **Image Processing** - Swap faces in static images
- **Video Processing** - Process entire video files with face swapping
- **Drag & Drop Support** - Easy file selection with drag-and-drop interface
- **Progress Tracking** - Real-time progress bars and status updates

### Advanced Features
- **Face Detection & Analysis** - Powered by InsightFace's Buffalo_L model
- **High-Quality Swapping** - Using InSwapper 128 ONNX model
- **Multi-face Support** - Detect and swap multiple faces simultaneously
- **Smart Face Masking** - Advanced masking for natural-looking results

---

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **GPU**: CUDA-compatible GPU recommended for better performance (optional)
- **Webcam**: Required for live face swapping features

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MIDHUNGRAJ/Deep-Face-Net.git
cd Deep-Face-Net
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> [!TIP]
> For **GPU acceleration**, install `onnxruntime-gpu` instead of `onnxruntime`.
> If you're on **CUDA 13** (common on Arch Linux / CachyOS), see the [Troubleshooting](#-troubleshooting) section below.

### 4. Download Models

The application requires several pre-trained models to function. Due to their large size, they are not included in the git repository.

You can **automatically download** all required models using the included script:

```bash
python download_models.py
```

This will download:
- **InSwapper 128** (Face Swapping)
- **GFPGAN** (Face Enhancement)

If the automated download fails, please refer to [models/README.md](models/README.md) for manual download links.

### 5. Optional: Virtual Camera Support

For virtual camera output (to use with OBS, Zoom, etc.):

```bash
pip install pyvirtualcam
```

**Linux users** also need to install v4l2loopback:
```bash
sudo apt-get install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1
```

---

## Usage

### GUI Mode (Default)

Launch the graphical interface:

```bash
python run.py
```

#### Live Camera Tab
1. Select a source image containing the face you want to use
2. Choose your camera from the dropdown
3. Click "Start Camera"
4. Click "Enable Face Swap" to begin swapping
5. Optional: Enable "Mouth Mask" or "Virtual Camera"

#### Offline Processing Tab
1. Select a source face image
2. Drag & drop or browse for a target video/image
3. Click "Start Processing"
4. Wait for completion and view/open the result

### CLI Mode

The CLI mode is activated when you provide command-line arguments.

#### Process a Video File

```bash
python run.py --source face.jpg --target video.mp4 --output result.mp4
```

#### Process an Image File

```bash
python run.py --source face.jpg --target photo.jpg --output result.jpg
```

#### Live Webcam Mode (CLI)

```bash
python run.py --source face.jpg --webcam
```

Use a specific camera:
```bash
python run.py --source face.jpg --webcam --camera-index 1
```

#### Additional Options

```bash
python run.py --source face.jpg --target video.mp4 --output result.mp4 --fps 30 --enhance
```

**Available Options:**
- `--source, -s`: Path to source image (face to swap from)
- `--target, -t`: Path to target image or video file
- `--output, -o`: Path to output file
- `--webcam`: Use webcam for live face swapping
- `--camera-index`: Camera index for webcam mode (default: 0)
- `--enhance`: Enable face enhancement (experimental)
- `--fps`: Target FPS for video processing (default: 30)

---

## Project Structure

```
Deep-Face-Net/
├── app/                          # GUI application
│   ├── deepfake_app.py          # Main Qt application
│   ├── video_thread.py          # Video processing thread
│   ├── file_processing_thread.py # File processing thread
│   └── drag_drop_widget.py      # Drag & drop widget
├── core/                         # Core functionality
│   ├── main.py                  # CLI entry point
│   ├── config.py                # Configuration settings
│   ├── face_analyser.py         # Face detection wrapper
│   ├── image_processor.py       # Image processing
│   ├── video_processor.py       # Video processing
│   └── engine/                  # Processing engines
│       ├── face_swapper.py      # Face swapping logic
│       ├── face_masking.py      # Face masking utilities
│       ├── face_enhancer.py     # Face enhancement
├── models/                       # Pre-trained models (not included)
├── testing/                      # Test files and samples
├── run.py                        # Main entry point
└── README.md                     # This file
```

---

## Configuration

Settings are stored in `core/config.py`:

- **Model Paths**: Configure paths to ONNX and enhancement models
- **Camera Settings**: Default camera index, resolution, and FPS
- **Face Detection**: Detection size and confidence threshold

User preferences (working directory, etc.) are saved in `~/.deepfacenet_settings.json`

---

## How It Works

1. **Face Detection**: Uses InsightFace's Buffalo_L model to detect faces in source and target
2. **Face Analysis**: Extracts facial landmarks and embeddings
3. **Face Swapping**: Uses InSwapper model to swap faces while preserving expressions
4. **Post-Processing**: Optional face enhancement and masking for natural results
5. **Output**: Renders the final result to screen, file, or virtual camera

---

## Limitations & Known Issues

- **GPU Acceleration**: ONNX Runtime with CUDA support recommended for real-time performance
- **Face Quality**: Works best with clear, front-facing faces
- **Lighting**: Significant lighting differences may affect quality
- **Multiple Faces**: Swaps all detected faces in the frame
- **Video Processing**: Large videos may take considerable time to process

---

## Troubleshooting

### GPU / CUDA 13 Support

The default `onnxruntime-gpu` package on PyPI may not support **CUDA 13** yet. If face swapping fails or falls back to CPU on systems running CUDA 13 (e.g., **Arch Linux / CachyOS**), install the nightly build instead:

```bash
# Install required dependencies
pip install coloredlogs flatbuffers numpy packaging protobuf sympy

# Install onnxruntime-gpu nightly with CUDA 13 support
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Ethical Considerations

**IMPORTANT**: This software is intended for educational and entertainment purposes only.

- **DO**: Use for learning, research, and creative projects
- **DO**: Obtain consent from individuals whose faces you use
- **DO**: Clearly label deepfake content when sharing
- **DON'T**: Create misleading or deceptive content
- **DON'T**: Use for harassment, fraud, or illegal activities
- **DON'T**: Violate anyone's privacy or rights

**Users are solely responsible for how they use this software. The developers assume no liability for misuse.**

---

## Acknowledgments

This project uses the following open-source technologies:

- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis and recognition
- [InSwapper](https://github.com/haofanwang/inswapper) - Face swapping model
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [ONNX Runtime](https://onnxruntime.ai/) - Model inference

---

## Contact

For questions, suggestions, or issues, please open an issue on GitHub.


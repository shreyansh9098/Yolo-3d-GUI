# YOLO-3D: Monocular 3D Object Detection System

![YOLO-3D Demo](assets/demo.gif)

YOLO-3D is an integrated computer vision system that combines YOLOv11 for object detection, Depth Anything v2 for depth estimation, and Segment Anything Model (SAM 2.0) for instance segmentation. This creates a comprehensive 3D scene understanding pipeline from a single camera input.

## Features

- **Multi-model Integration**: Combines state-of-the-art models for detection, depth, and segmentation
- **3D Visualization**: Renders 3D bounding boxes with accurate depth perception
- **Bird's Eye View**: Top-down spatial visualization of detected objects
- **Instance Segmentation**: Precise object masks using SAM 2.0
- **Object Tracking**: Track objects across video frames
- **Modern GUI**: User-friendly interface for video processing and visualization
- **Multi-view Display**: View original, detection, depth, segmentation, and 3D results simultaneously

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA compatible GPU recommended (though CPU mode works)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   https://github.com/Pavankunchala/Yolo-3d-GUI.git
   cd YOLO-3d-GUI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Models will be downloaded automatically on first run or you can download them manually:
   ```bash
   # YOLOv11 Nano model
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt
   
   # SAM 2.0 Base model
   wget https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_b.pt
   ```

## Usage

### GUI Interface

The easiest way to use YOLO-3D is through the GUI:

```bash
python yolo3d_gui.py
```

This will open the interface where you can:
- Select video files or webcam input
- Choose models and adjust parameters
- Toggle different features
- View all visualization modes
- Save processed video or individual frames

### Command Line

For batch processing or integration into other pipelines, you can use the command line:

```bash
python run.py --source path/to/video.mp4 --output output.mp4 --yolo nano --depth small --sam sam2_b.pt
```

### API Usage

You can also use the components in your own Python code:

```python
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from segmentation_model import SegmentationModel
from bbox3d_utils import BBox3DEstimator, BirdEyeView

# Initialize models
detector = ObjectDetector(model_size="nano")
depth_estimator = DepthEstimator(model_size="small")
segmenter = SegmentationModel(model_name="sam2_b.pt")
bbox3d_estimator = BBox3DEstimator()
bev = BirdEyeView(scale=60, size=(300, 300))

# Process a frame
frame = cv2.imread("example.jpg")
detections = detector.detect(frame)
depth_map = depth_estimator.estimate_depth(frame)
segmentation = segmenter.segment_with_boxes(frame, [d[0] for d in detections])

# Create 3D visualization
for detection, seg_result in zip(detections, segmentation):
    bbox, score, class_id, obj_id = detection
    depth_value = depth_estimator.get_depth_in_region(depth_map, bbox)
    box_3d = {'bbox_2d': bbox, 'depth_value': depth_value, 'class_name': detector.get_class_names()[class_id], 'mask': seg_result['mask']}
    frame = bbox3d_estimator.draw_box_3d(frame, box_3d)
```

## Project Structure

```
YOLO-3D/
│
├── yolo3d_gui.py           # GUI application
├── run.py                  # Command line entry point
├── detection_model.py      # YOLOv11 detector implementation
├── depth_model.py          # Depth Anything v2 implementation
├── segmentation_model.py   # SAM 2.0 implementation
├── bbox3d_utils.py         # 3D bounding box and BEV utilities
├── requirements.txt        # Project dependencies
├── assets/                 # Demo images/videos
└── README.md               # This file
```

## Performance Optimization

For better performance:

- Use smaller models (nano/small) for real-time applications
- Enable frame skipping for segmentation (every 2-3 frames)
- Process at a lower resolution (640x480) for faster inference
- Use CUDA GPU for acceleration (10-20x faster than CPU)
- Consider batch processing for offline video analysis

## Acknowledgments
- [Yolo-3d](https://github.com/niconielsen32/YOLO-3D) for the initial scripts

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11 and SAM implementations
- [Depth Anything](https://github.com/LiheYoung/Depth-Anything) for the depth estimation model
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) for the GUI framework

## License

This project is released under the MIT License. See the LICENSE file for details.

---

If you find this project useful, consider giving it a star! For issues, feature requests, or contributions, please open an issue or pull request.

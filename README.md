# Vehicle and License Plate Detection System
![Python](https://img.shields.io/badge/python-v3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A real-time computer vision system for vehicle detection, license plate recognition, and tracking using YOLOv8 and EasyOCR.



https://github.com/user-attachments/assets/c66ed79a-effa-424e-8d49-55c6a18eee94







## Features

### Real-time Detection & Tracking
- Vehicle detection using YOLOv8 or YOLO11
- License plate recognition with EasyOCR
- Real-time vehicle tracking using SORT algorithm
- Customizable detection zone
- Multi-vehicle simultaneous tracking

### Advanced Analytics
- License plate text extraction
- Vehicle tracking statistics
- Detection confidence scores
- Frame-by-frame analysis
- Results export to CSV

### Modern Interface
- High-resolution video preview (1920x1080)
- Real-time detection visualization
- Detection zone display
- Exit instructions overlay
- Progress tracking

### Visual Indicators
- Green corner borders for vehicles
- Red rectangles for license plates
- Yellow detection zone
- White-background text displays
- Dynamic size adjustments

## Setup

### Clone the repository:
```bash
git clone https://github.com/mumutozbek/Vehicle_Plate_Detection.git
cd Vehicle_Plate_Detection
```

### Create and activate virtual environment:
```bash
# For macOS/Linux
python3.10 -m venv venv310
source venv310/bin/activate

# For Windows
python3.10 -m venv venv310
venv310\Scripts\activate
```

### Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Start the application:
```bash
python main.py
```

### System Operation:
1. Place input video in project directory
2. System automatically downloads YOLOv8 if needed
3. Monitor real-time detections
4. Press 'q' to exit and save results

## Project Structure
```
license-plate-detection/
├── main.py              # Core detection script
├── util.py             # Utility functions
├── sort/               # Tracking algorithm
│   └── sort.py
├── models/             # YOLO models
│   ├── yolov8x.pt
│   └── license_plate_detector.pt
└── requirements.txt    # Dependencies
```

## Key Features Details

### License Plate Detection
- Real-time plate recognition
- Text extraction and processing
- Confidence score calculation
- Multi-plate handling

### Vehicle Tracking
- Multiple vehicle class detection
  - Cars (class 2)
  - Motorcycles (class 3)
  - Buses (class 5)
  - Trucks (class 7)
- Continuous tracking with SORT
- Detection zone filtering

### Analytics Dashboard
- Frame number tracking
- Vehicle IDs
- Bounding box coordinates
- License plate text
- Confidence scores

### Video Processing
- High-resolution output (1920x1080)
- Original frame rate preservation
- Automatic video saving
- CSV result export

## Requirements

### Hardware
- Minimum 8GB RAM
- CUDA-capable GPU (recommended)

### Software
```
numpy==1.24.3
opencv-python==4.7.0.72
torch==2.0.1
torchvision==0.15.2
ultralytics==8.0.114
pandas==2.0.2
easyocr==1.6.2
filterpy==1.4.5
scipy==1.10.1
Pillow==9.5.0
```

## Output

### Video Output
- Filename: output_detection.mp4
- Resolution: 1920x1080
- Format: MP4 (mp4v codec)
- Frame rate: Matches input

### Data Export
- Filename: test.csv
- Format: CSV
- Contents: Frame numbers, vehicle IDs, coordinates, license plates

## Author
Mustafa Umut Ozbek

## License
MIT License

## Acknowledgments
- [Ultralytics YOLOv8](https://github.com/ultralytics/yolov8) or yolo11
- [SORT Algorithm](https://github.com/abewley/sort)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

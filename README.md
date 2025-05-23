# Golf Swing Analyzer

A simple Python application that uses MediaPipe to analyze golf swings from video footage. The analyzer tracks key body landmarks and provides measurements such as spine angle and knee flex throughout the swing.

## Features

- Pose detection using MediaPipe
- Real-time body landmark tracking
- Spine angle measurement
- Knee flex angle measurement
- Visual overlay of measurements on the video

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your golf swing video file in the project directory
2. Update the input and output video paths in `golf_swing_analyzer.py`:
```python
input_video = "your_swing_video.mp4"
output_video = "analyzed_swing.mp4"
```
3. Run the script:
```bash
python golf_swing_analyzer.py
```

## Measurements

The analyzer provides the following measurements:
- **Spine Angle**: The angle of the spine relative to vertical
- **Knee Flex**: The angle of the left knee bend

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Notes

- For best results, ensure the golfer is fully visible in the video
- Record the video from a side view for optimal angle measurements
- Good lighting conditions will improve pose detection accuracy 
# First Cut — Motion-Controlled Samurai

A motion-controlled external controller for [First Cut](https://drasnus.itch.io/first-cut), built using computer vision and smartphone motion sensors.

The project explores controlling an existing game using real-world body movements and sword motions, without modifying the original game.

## Overview

What if you could control a samurai using your own movements?

This project combines webcam-based pose tracking with smartphone IMU data to translate physical movements into keyboard inputs for First Cut.

The webcam tracks body movement, while a smartphone provides acceleration, gravity, and orientation data through **phyphox**. Python processes these inputs and maps recognized movements to the game's controls.

## Features

- **Body Movement Tracking:** Uses MediaPipe Pose Landmarker to track body landmarks through a webcam.
- **Motion Detection:** Uses smartphone accelerometer and gravity data to distinguish sword movements.
- **Directional Attacks:** Detects horizontal and vertical sword swings.
- **Guard Positioning:** Uses smartphone orientation to determine guard positioning.
- **Real-Time Control:** Translates detected movements into keyboard inputs using Python.
- **Threshold-Based Detection:** Uses experimentally tuned thresholds to distinguish movements.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main application logic |
| OpenCV | Webcam input and image processing |
| MediaPipe | Real-time body pose tracking |
| phyphox | Smartphone sensor data acquisition |
| Requests | Communication with phyphox |
| pynput | Keyboard input simulation |

## How It Works

The controller combines two independent input sources:

**1. Computer Vision**

- Captures webcam frames using OpenCV.
- Processes frames using MediaPipe Pose Landmarker.
- Tracks relevant body landmarks.
- Uses body movement to determine player movement.

**2. Smartphone IMU**

- Uses phyphox to access smartphone sensor readings.
- Collects acceleration, gravity, and orientation data.
- Processes sensor values to identify sword movements.
- Applies threshold-based rules to classify movement directions and actions.

**3. Input Mapping**

The detected movements are translated into keyboard inputs using pynput, allowing the player to control First Cut without modifying the original game.

## Controls

| Physical Movement | Game Action |
|---|---|
| Body movement to the left/right | Player movement |
| Horizontal sword swing | Horizontal attack |
| Up-to-down sword swing | High vertical attack |
| Down-to-up sword swing | Low vertical attack |
| Smartphone orientation | Guard positioning |

The controller also maps the game's existing keyboard controls for actions such as dodging and shoving.

## Threshold Calibration

One of the important parts of this project was determining how real-world movements translate into reliable sensor readings.

Rather than choosing arbitrary values, thresholds were tuned through repeated movement trials:

1. Collect sensor readings while performing different movements.
2. Observe how acceleration and gravity values change.
3. Compare readings across different movement types.
4. Adjust thresholds based on observed differences.
5. Test the resulting detection rules in the game.

The final tuned values are stored in `data/thresholds.json`.

These values are experimental and may require recalibration depending on the smartphone, its orientation, and the user's movement.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version for your MediaPipe installation.
- A webcam.
- A smartphone with phyphox installed.
- First Cut running on your computer.

### Installation

Clone the repository:

```bash
git clone https://github.com/rithvikkesarla-hub/First-Cut-Motion-Controller.git
cd First-Cut-Motion-Controller
```

Install the dependencies:

```bash
pip install opencv-python mediapipe requests pynput
```

Download the MediaPipe Pose Landmarker Lite model:

[Download pose_landmarker_lite.task](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)

Place the downloaded model in the project root directory.

### Smartphone Setup

1. Install phyphox on your smartphone.
2. Configure the required sensor experiment.
3. Enable phyphox's remote access.
4. Connect your smartphone and computer to the same network.
5. Configure the remote access URL in the controller script.

### Run the Controller

```bash
python "working movement + swings.py"
```

Start First Cut and test the controller.

## Project Structure

```text
First-Cut-Motion-Controller/
│
├── working movement + swings.py
│
├── data/
│   └── thresholds.json
│
├── .gitignore
│
└── README.md
```

The MediaPipe model is downloaded separately and is not included in the repository.

## Current Limitations

- Sensor readings can vary between smartphones.
- Threshold-based detection depends on calibration and movement consistency.
- Webcam-based tracking depends on lighting, camera positioning, and visibility.
- The controller is designed around the tested setup and may require adjustments on other systems.

## Credits

- [First Cut — Drasnus](https://drasnus.itch.io/first-cut)
- [MediaPipe](https://ai.google.dev/edge/mediapipe)
- [OpenCV](https://opencv.org/)
- [phyphox](https://phyphox.org/)

---

*An experimental project I hope to showcase at an event, inspiring others to explore new technologies, learn something new, and build something innovative of their own.*

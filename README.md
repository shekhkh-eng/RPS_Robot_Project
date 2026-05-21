# RPS Robot Project

Real-time **Rock Paper Scissors** game where you play against a robot using your webcam. The app detects your hand with a **pre-trained neural network** (MediaPipe Hand Landmarker), classifies your gesture, and the robot picks a random move.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)

## Features

- Live webcam hand tracking with landmark overlay
- Gesture recognition: **ROCK**, **PAPER**, **SCISSORS**
- Random robot opponent
- Winner logic and score tracking (player / robot / draws)
- Pygame GUI with move images and status panel
- Press **SPACE** to start a round (3-2-1 countdown)

## Project structure

```
RPS_Robot_Project/
├── assets/
│   ├── hand_landmarker.task   # Pre-trained MediaPipe model
│   ├── rock.png
│   ├── paper.png
│   ├── scissors.png
│   └── robot.png
├── screenshots/
├── sounds/
├── game_logic.py      # Rules, scoring, robot moves
├── hand_detector.py   # MediaPipe + gesture classification
├── main.py            # Game loop and webcam
├── ui.py              # Pygame interface
├── requirements.txt
└── README.md
```

## Installation

**Requirements:** Python 3.11, webcam

1. Clone or download this project and open a terminal in the project folder.

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   ```

   Windows:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   macOS / Linux:

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the AI model file exists:

   - Path: `assets/hand_landmarker.task`
   - If missing, download:

     ```text
     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
     ```

     Save the file as `assets/hand_landmarker.task`.

## How to run

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| **SPACE** | Start a round (3-2-1 countdown, then lock your gesture) |
| **R** | Reset scores |
| **Q** or **ESC** | Quit |

### How to play

1. Stand where your hand is visible in the camera.
2. When ready, press **SPACE**.
3. During the countdown, show:
   - **ROCK** — closed fist (0–1 fingers)
   - **PAPER** — open palm (4–5 fingers)
   - **SCISSORS** — index + middle only (2 fingers)
4. After the countdown, your gesture is locked and the robot plays.
5. The winner and updated score appear on the right panel.

**Tips:** Use good lighting, keep your hand inside the frame, and hold the gesture steady during the countdown.

## AI model explanation

This project uses **two layers** of intelligence:

### 1. Pre-trained neural network (MediaPipe Hand Landmarker)

The file `assets/hand_landmarker.task` is a **TensorFlow Lite** model trained by Google. It runs on every camera frame and:

- Detects the hand in the image
- Predicts **21 3D landmarks** (knuckles, fingertips, etc.)

That model is a real-time deep learning pipeline (palm detection + hand landmark regression). You do not train it in this project; it ships pre-trained.

### 2. Rule-based gesture classifier (finger counting)

From the landmarks, the app counts how many fingers are extended:

| Fingers | Gesture |
|---------|---------|
| 0–1 | ROCK |
| 2 | SCISSORS |
| 4–5 | PAPER |
| 3 | Unclear (hold a clearer pose) |

Results are **smoothed** over several frames to reduce flicker. The robot move is **random** (`random.choice`); the “AI” opponent does not learn from past games.

## Technologies used

| Technology | Role |
|------------|------|
| **Python 3.11** | Main language |
| **OpenCV** | Webcam capture and image processing |
| **MediaPipe** | Pre-trained hand landmark model (Tasks API) |
| **NumPy** | Array handling for frames |
| **Pygame** | GUI, layout, and input |
| **Pillow** | Asset image support (optional generation) |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Could not open webcam` | Close other apps using the camera; try another USB port |
| `Hand landmarker model not found` | Download `hand_landmarker.task` into `assets/` |
| Gesture not detected | Improve lighting; show full hand; avoid 3-finger poses |
| Wrong gesture | Adjust hand angle; only index+middle for scissors |

## License

Educational / portfolio project. MediaPipe models are subject to [Google's MediaPipe license](https://github.com/google-ai-edge/mediapipe).

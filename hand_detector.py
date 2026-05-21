"""
Hand gesture detection using MediaPipe Hand Landmarker (pre-trained TFLite model).
Maps finger counts to ROCK, PAPER, or SCISSORS.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Gesture labels returned to the game
ROCK = "ROCK"
PAPER = "PAPER"
SCISSORS = "SCISSORS"

# MediaPipe hand skeleton connections for drawing
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class HandDetector:
    """Detects hands with MediaPipe and classifies RPS gestures from landmarks."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        smooth_window: int = 8,
        smooth_threshold: int = 5,
    ) -> None:
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "assets", "hand_landmarker.task")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at '{model_path}'. "
                "Download hand_landmarker.task into the assets folder."
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        self._gesture_history: Deque[str] = deque(maxlen=smooth_window)
        self._smooth_threshold = smooth_threshold
        self._last_raw_gesture: Optional[str] = None
        self._finger_count: int = 0

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def process(
        self, frame: np.ndarray
    ) -> Tuple[Optional[str], np.ndarray, int]:
        """
        Detect hand, draw landmarks, and return smoothed gesture.

        Returns:
            (gesture or None, annotated frame, finger count)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        gesture: Optional[str] = None
        self._finger_count = 0

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            self._draw_landmarks(frame, landmarks)
            self._finger_count = self._count_fingers(landmarks)
            raw_gesture = self._fingers_to_gesture(self._finger_count)
            self._last_raw_gesture = raw_gesture

            if raw_gesture:
                self._gesture_history.append(raw_gesture)
                gesture = self._get_smoothed_gesture()
        else:
            self._gesture_history.clear()
            self._last_raw_gesture = None

        return gesture, frame, self._finger_count

    def get_last_raw_gesture(self) -> Optional[str]:
        """Most recent single-frame gesture (before smoothing)."""
        return self._last_raw_gesture

    def reset_smoothing(self) -> None:
        """Clear gesture history (e.g. after a round ends)."""
        self._gesture_history.clear()

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, landmarks) -> None:
        h, w, _ = frame.shape
        points = [
            (int(lm.x * w), int(lm.y * h))
            for lm in landmarks
        ]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 128, 255), -1)

    @staticmethod
    def _count_fingers(landmarks) -> int:
        """Count extended fingers using landmark positions."""
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]

        fingers_up = 0
        wrist = landmarks[0]

        # Thumb: tip farther from wrist than the IP joint (works for mirror flip)
        thumb_tip_dist = abs(landmarks[4].x - wrist.x) + abs(landmarks[4].y - wrist.y)
        thumb_ip_dist = abs(landmarks[3].x - wrist.x) + abs(landmarks[3].y - wrist.y)
        if thumb_tip_dist > thumb_ip_dist * 1.05:
            fingers_up += 1

        # Other four fingers: tip above pip (smaller y in image coordinates)
        for tip_id, pip_id in zip(tips[1:], pips[1:]):
            if landmarks[tip_id].y < landmarks[pip_id].y:
                fingers_up += 1

        return fingers_up

    @staticmethod
    def _fingers_to_gesture(finger_count: int) -> Optional[str]:
        """
        Map finger count to RPS gesture.
        0-1: fist -> ROCK, 2: index+middle -> SCISSORS, 4-5: open palm -> PAPER.
        """
        if finger_count <= 1:
            return ROCK
        if finger_count == 2:
            return SCISSORS
        if finger_count >= 4:
            return PAPER
        return None  # 3 fingers is ambiguous

    def _get_smoothed_gesture(self) -> Optional[str]:
        if len(self._gesture_history) < self._smooth_threshold:
            return None

        counts: dict[str, int] = {}
        for g in self._gesture_history:
            counts[g] = counts.get(g, 0) + 1

        best_gesture, best_count = max(counts.items(), key=lambda item: item[1])
        if best_count >= self._smooth_threshold:
            return best_gesture
        return None

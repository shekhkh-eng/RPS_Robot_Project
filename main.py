"""
RPS Robot Project — Real-time Rock Paper Scissors against an AI robot.
Uses webcam, MediaPipe hand detection, and a Pygame GUI.
"""

from __future__ import annotations

import sys
import time

import cv2

from game_logic import GameLogic, PAPER, ROCK, SCISSORS
from hand_detector import HandDetector
from ui import GameUI

# How long to show round results before clearing move display
RESULT_DISPLAY_SEC = 2.5
COUNTDOWN_SEC = 3


class RPSRobotGame:
    """Main game controller: webcam loop, gestures, rounds, and UI."""

    def __init__(self, camera_index: int = 0) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check that a camera is connected and not in use."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.detector = HandDetector()
        self.logic = GameLogic()
        self.ui = GameUI()

        self.running = True
        self._in_countdown = False
        self._countdown_end = 0.0
        self._locked_gesture: str | None = None
        self._show_result_until = 0.0
        self._display_player: str | None = None
        self._display_robot: str | None = None
        self._display_message: str | None = None

    def run(self) -> None:
        """Main game loop."""
        self.ui.set_status("Show ROCK / PAPER / SCISSORS — press SPACE to play")

        while self.running:
            action = self.ui.handle_events()
            if action == "quit":
                self.running = False
                break
            if action == "reset":
                self.logic.reset_scores()
                self._clear_round_display()
                self.ui.set_status("Scores reset — press SPACE to play")
            if action == "play" and not self._in_countdown:
                self._start_countdown()

            success, frame = self.cap.read()
            if not success:
                self.ui.set_status("Camera read failed — check your webcam")
                self.ui.tick()
                continue

            # Mirror frame so movement matches the player's view
            frame = cv2.flip(frame, 1)
            gesture, frame, finger_count = self.detector.process(frame)
            self._update_countdown(gesture)

            # After result period, clear displayed moves but keep scores
            if self._display_message and time.time() > self._show_result_until:
                self._display_message = None
                self._display_player = None
                self._display_robot = None

            self.ui.render(
                frame=frame,
                player_move=self._display_player,
                robot_move=self._display_robot,
                result_message=self._display_message,
                score_text=self.logic.get_score_text(),
                finger_count=finger_count,
                detected_gesture=gesture,
            )
            self.ui.tick(30)

        self.cleanup()

    def _start_countdown(self) -> None:
        """Begin 3-2-1 countdown before locking the player's gesture."""
        self._in_countdown = True
        self._countdown_end = time.time() + COUNTDOWN_SEC
        self._locked_gesture = None
        self._display_message = None
        self.detector.reset_smoothing()
        self.ui.set_status("Get ready...")

    def _update_countdown(self, current_gesture: str | None) -> None:
        if not self._in_countdown:
            self.ui.set_countdown(None)
            return

        remaining = self._countdown_end - time.time()
        if remaining > 0:
            # Show 3, 2, 1 on screen
            self.ui.set_countdown(max(1, int(remaining + 0.99)))
            # Keep updating best gesture during countdown
            if current_gesture:
                self._locked_gesture = current_gesture
        else:
            self.ui.set_countdown(None)
            self._in_countdown = False
            self._finish_round()

    def _finish_round(self) -> None:
        """Lock player move, get robot move, and update scores."""
        player_move = self._locked_gesture

        if player_move not in (ROCK, PAPER, SCISSORS):
            self.ui.set_status(
                "No clear gesture — show ROCK (fist), PAPER (open hand), "
                "or SCISSORS (2 fingers)"
            )
            self.detector.reset_smoothing()
            return

        result = self.logic.play_round(player_move)
        self._display_player = result.player_move
        self._display_robot = result.robot_move
        self._display_message = result.message
        self._show_result_until = time.time() + RESULT_DISPLAY_SEC

        self.ui.set_status(f"Round {self.logic.rounds_played} complete — SPACE for next")
        self.detector.reset_smoothing()

    def _clear_round_display(self) -> None:
        self._display_player = None
        self._display_robot = None
        self._display_message = None
        self._in_countdown = False
        self.ui.set_countdown(None)

    def cleanup(self) -> None:
        self.detector.close()
        self.cap.release()
        cv2.destroyAllWindows()
        self.ui.quit()


def main() -> None:
    try:
        game = RPSRobotGame(camera_index=0)
        game.run()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        print(
            "Download the model:\n"
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task\n"
            "Save it as assets/hand_landmarker.task",
            file=sys.stderr,
        )
        sys.exit(1)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

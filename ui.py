"""
Pygame GUI for the RPS Robot Project.
Displays camera feed, moves, winner, score, and asset images.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np
import pygame

# Layout constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PANEL_X = 660
PANEL_WIDTH = WINDOW_WIDTH - PANEL_X - 20

# Colors
COLOR_BG = (18, 22, 32)
COLOR_PANEL = (28, 34, 48)
COLOR_TEXT = (240, 240, 245)
COLOR_ACCENT = (80, 180, 255)
COLOR_WIN = (100, 230, 140)
COLOR_LOSE = (255, 110, 110)
COLOR_DRAW = (255, 210, 80)
COLOR_HINT = (160, 170, 190)


class GameUI:
    """Renders the game window with camera view and status panel."""

    def __init__(self, assets_dir: Optional[str] = None) -> None:
        pygame.init()
        pygame.display.set_caption("RPS Robot Project — Rock Paper Scissors AI")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        if assets_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(base_dir, "assets")

        self._assets_dir = assets_dir
        self._move_images: Dict[str, pygame.Surface] = {}

        self._font_title = pygame.font.SysFont("Segoe UI", 32, bold=True)
        self._font_large = pygame.font.SysFont("Segoe UI", 26, bold=True)
        self._font_medium = pygame.font.SysFont("Segoe UI", 22)
        self._font_small = pygame.font.SysFont("Consolas", 18)
        self._load_assets()

        self.countdown_value: Optional[int] = None
        self.status_message = "Show your hand — press SPACE to play"

    def _load_assets(self) -> None:
        """Load gesture images from the assets folder."""
        mapping = {
            "ROCK": "rock.png",
            "PAPER": "paper.png",
            "SCISSORS": "scissors.png",
            "ROBOT": "robot.png",
        }
        for key, filename in mapping.items():
            path = os.path.join(self._assets_dir, filename)
            if os.path.isfile(path):
                img = pygame.image.load(path).convert_alpha()
                self._move_images[key] = pygame.transform.smoothscale(img, (140, 140))
            else:
                # Fallback placeholder if image is missing
                surf = pygame.Surface((140, 140), pygame.SRCALPHA)
                surf.fill((50, 55, 70))
                pygame.draw.rect(surf, COLOR_ACCENT, surf.get_rect(), 3, border_radius=12)
                label = self._font_small.render(key[:3], True, COLOR_TEXT)
                surf.blit(label, label.get_rect(center=(70, 70)))
                self._move_images[key] = surf

    def set_countdown(self, value: Optional[int]) -> None:
        """Show 3-2-1 countdown overlay on camera (None to hide)."""
        self.countdown_value = value

    def set_status(self, message: str) -> None:
        self.status_message = message

    def handle_events(self) -> Optional[str]:
        """
        Process pygame events.

        Returns:
            'quit', 'play', 'reset', or None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return "quit"
                if event.key == pygame.K_SPACE:
                    return "play"
                if event.key == pygame.K_r:
                    return "reset"
        return None

    def draw_camera(self, frame: np.ndarray) -> None:
        """Draw the OpenCV BGR frame on the left side of the window."""
        resized = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(rgb.tobytes(), (CAMERA_WIDTH, CAMERA_HEIGHT), "RGB")
        self.screen.blit(surf, (20, 100))

        # Camera border
        pygame.draw.rect(
            self.screen, COLOR_ACCENT,
            pygame.Rect(18, 98, CAMERA_WIDTH + 4, CAMERA_HEIGHT + 4), 2, border_radius=8
        )

        # Countdown overlay
        if self.countdown_value is not None:
            overlay = pygame.Surface((CAMERA_WIDTH, CAMERA_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (20, 100))
            text = self._font_title.render(str(self.countdown_value), True, (255, 255, 100))
            rect = text.get_rect(center=(20 + CAMERA_WIDTH // 2, 100 + CAMERA_HEIGHT // 2))
            self.screen.blit(text, rect)

    def draw_panel(
        self,
        player_move: Optional[str],
        robot_move: Optional[str],
        result_message: Optional[str],
        score_text: str,
        finger_count: int = 0,
        detected_gesture: Optional[str] = None,
    ) -> None:
        """Draw the right-side info panel."""
        panel_rect = pygame.Rect(PANEL_X, 90, PANEL_WIDTH, WINDOW_HEIGHT - 110)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_ACCENT, panel_rect, 2, border_radius=12)

        y = 110
        title = self._font_title.render("Battle", True, COLOR_TEXT)
        self.screen.blit(title, (PANEL_X + 20, y))
        y += 50

        # Player vs Robot images
        self._draw_move_row(PANEL_X + 30, y, "You", player_move)
        self._draw_move_row(PANEL_X + 30, y + 180, "Robot", robot_move, is_robot=True)
        y += 370

        # Winner / status
        if result_message:
            if "You Win" in result_message:
                color = COLOR_WIN
            elif "Robot Wins" in result_message:
                color = COLOR_LOSE
            else:
                color = COLOR_DRAW
            msg_surf = self._font_large.render(result_message, True, color)
        else:
            msg_surf = self._font_medium.render(self.status_message, True, COLOR_HINT)
        self.screen.blit(msg_surf, (PANEL_X + 20, y))
        y += 45

        # Score
        score_surf = self._font_medium.render(score_text, True, COLOR_ACCENT)
        self.screen.blit(score_surf, (PANEL_X + 20, y))
        y += 40

        # Live detection hint
        hint = f"Fingers: {finger_count}"
        if detected_gesture:
            hint += f"  |  Detected: {detected_gesture}"
        hint_surf = self._font_small.render(hint, True, COLOR_HINT)
        self.screen.blit(hint_surf, (PANEL_X + 20, y))

    def _draw_move_row(
        self,
        x: int,
        y: int,
        label: str,
        move: Optional[str],
        is_robot: bool = False,
    ) -> None:
        label_surf = self._font_medium.render(label, True, COLOR_TEXT)
        self.screen.blit(label_surf, (x, y))

        img_key = "ROBOT" if is_robot and move is None else (move or "")
        if img_key in self._move_images:
            self.screen.blit(self._move_images[img_key], (x + 120, y - 10))
        else:
            placeholder = pygame.Surface((140, 140))
            placeholder.fill((40, 45, 58))
            pygame.draw.rect(placeholder, COLOR_HINT, placeholder.get_rect(), 2)
            self.screen.blit(placeholder, (x + 120, y - 10))

        move_text = move if move else "—"
        move_surf = self._font_large.render(move_text, True, COLOR_ACCENT)
        self.screen.blit(move_surf, (x + 280, y + 50))

    def draw_header(self) -> None:
        """Draw title bar and controls."""
        header = self._font_title.render("RPS Robot Project", True, COLOR_TEXT)
        self.screen.blit(header, (20, 20))
        controls = self._font_small.render(
            "SPACE: Play round  |  R: Reset score  |  Q: Quit",
            True, COLOR_HINT,
        )
        self.screen.blit(controls, (20, 58))

    def render(
        self,
        frame: np.ndarray,
        player_move: Optional[str] = None,
        robot_move: Optional[str] = None,
        result_message: Optional[str] = None,
        score_text: str = "",
        finger_count: int = 0,
        detected_gesture: Optional[str] = None,
    ) -> None:
        """Full frame render."""
        self.screen.fill(COLOR_BG)
        self.draw_header()
        self.draw_camera(frame)
        self.draw_panel(
            player_move, robot_move, result_message, score_text,
            finger_count, detected_gesture,
        )
        pygame.display.flip()

    def tick(self, fps: int = 30) -> None:
        self.clock.tick(fps)

    def quit(self) -> None:
        pygame.quit()

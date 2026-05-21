"""
Rock Paper Scissors game rules, scoring, and robot opponent logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

ROCK = "ROCK"
PAPER = "PAPER"
SCISSORS = "SCISSORS"

ALL_MOVES = (ROCK, PAPER, SCISSORS)

# Who wins: (player_move, robot_move) -> "player" | "robot" | "draw"
_WIN_TABLE = {
    (ROCK, SCISSORS): "player",
    (SCISSORS, PAPER): "player",
    (PAPER, ROCK): "player",
    (SCISSORS, ROCK): "robot",
    (PAPER, SCISSORS): "robot",
    (ROCK, PAPER): "robot",
}


@dataclass
class RoundResult:
    """Outcome of a single round."""

    player_move: str
    robot_move: str
    winner: str  # "player", "robot", or "draw"
    message: str


class GameLogic:
    """Manages random robot moves, winner determination, and score tracking."""

    def __init__(self) -> None:
        self.player_score = 0
        self.robot_score = 0
        self.draws = 0
        self.rounds_played = 0
        self.last_result: Optional[RoundResult] = None

    def get_robot_move(self) -> str:
        """Robot picks a random move."""
        return random.choice(ALL_MOVES)

    @staticmethod
    def determine_winner(player_move: str, robot_move: str) -> str:
        """
        Return winner key: 'player', 'robot', or 'draw'.
        """
        if player_move == robot_move:
            return "draw"
        return _WIN_TABLE.get((player_move, robot_move), "draw")

    def play_round(self, player_move: str) -> RoundResult:
        """Play one round, update scores, and return the result."""
        robot_move = self.get_robot_move()
        winner = self.determine_winner(player_move, robot_move)

        if winner == "player":
            self.player_score += 1
            message = "You Win!"
        elif winner == "robot":
            self.robot_score += 1
            message = "Robot Wins!"
        else:
            self.draws += 1
            message = "Draw!"

        self.rounds_played += 1
        result = RoundResult(
            player_move=player_move,
            robot_move=robot_move,
            winner=winner,
            message=message,
        )
        self.last_result = result
        return result

    def reset_scores(self) -> None:
        """Reset all scores to zero."""
        self.player_score = 0
        self.robot_score = 0
        self.draws = 0
        self.rounds_played = 0
        self.last_result = None

    def get_score_text(self) -> str:
        return (
            f"Player: {self.player_score}  |  "
            f"Robot: {self.robot_score}  |  "
            f"Draws: {self.draws}"
        )

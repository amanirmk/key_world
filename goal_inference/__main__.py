import typing
from goal_inference.game import Game
from goal_inference.replay import Replay
from goal_inference.world import generate_worlds
import argparse
import os
import pathlib


def record_game(args):
    pass


def evaluate_data(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--subj", type=str, default=None)
    parser.add_argument("--folder", type=str, default="game_logs")
    args = parser.parse_args()

    assert (args.play or args.eval) and not (args.play and args.eval)
    if args.play:
        assert args.subj is not None
        record_game(args)
    else:
        evaluate_data(args)

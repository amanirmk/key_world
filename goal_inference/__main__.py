from goal_inference.game import Game
from goal_inference.replay import Replay
from goal_inference.world import all_worlds
import argparse
import pathlib
import numpy as np
import os
from tqdm import tqdm  # type: ignore[import-untyped]
import gc


def hyperparam_search():
    alphas = np.logspace(0, 5, num=6, base=4)
    n_turns = np.arange(1, 5 + 1)
    p_actions = np.linspace(0.01, 0.65, num=5)
    p_goals = np.linspace(0.01, 0.65, num=5)
    for alpha in alphas:
        for n in n_turns:
            yield alpha, ("turn", n)
        for p in p_actions:
            yield alpha, ("action", p)
        for p in p_goals:
            yield alpha, ("goal", p)


def record_game(args):
    assert args.subj is not None, "Must provide a unique subject identifier"
    folder = pathlib.Path(args.record_game_folder)
    folder.mkdir(parents=True, exist_ok=True)
    played_worlds = [
        int(f.stem.split("_")[1])
        for f in folder.iterdir()
        if f.stem.startswith(args.subj)
    ]
    assert len(played_worlds) == len(
        set(played_worlds)
    ), "Uh oh, there is a duplicate world!"
    print(f"Worlds explored: {len(played_worlds)}")
    assert len(played_worlds) <= len(all_worlds), "You have already played all worlds!"
    remaining_world_idxs = list(set(range(len(all_worlds))) - set(played_worlds))
    world_idx = np.random.choice(remaining_world_idxs)
    world = all_worlds[world_idx]
    game = Game(
        world=world,
        human_player=True,
        record=True,
        output_folder=folder,
        csv_name=f"{args.subj}_{world_idx}.csv",
        alpha=-1,
        update_criteria=("None", -1),
        gui=True,
    )
    game.play()


def run_model(args):
    folder = pathlib.Path(args.run_model_folder)
    folder.mkdir(parents=True, exist_ok=True)
    for alpha, update_criteria in tqdm(hyperparam_search(), total=90):
        for idx, world in enumerate(all_worlds):
            game = Game(
                world=world,
                human_player=False,
                record=True,
                output_folder=folder,
                csv_name=f"alpha={alpha}_update={update_criteria}_{idx}.csv",
                alpha=alpha,
                update_criteria=update_criteria,
                gui=False,
            )
            game.play()
            del game
            gc.collect()


def evaluate_data(args):
    output_folder = pathlib.Path(args.eval_data_folder)
    input_folder = pathlib.Path(args.record_game_folder)
    assert os.path.exists(input_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for alpha, update_criteria in tqdm(hyperparam_search(), total=90):
        for human_csv in input_folder.iterdir():
            idx = int(human_csv.stem.split("_")[1])
            replay = Replay(
                world=all_worlds[idx],
                human_csv=human_csv,
                alpha=alpha,
                update_criteria=update_criteria,
            )
            data = replay.replay()
            output_file = (
                human_csv.stem + f"_alpha={alpha}_update={update_criteria}.csv"
            )
            data.to_csv(output_folder / output_file, index=False)
            del replay
            del data
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_game_folder", type=str, default="human_data")
    parser.add_argument("--run_model_folder", type=str, default="model_data")
    parser.add_argument("--eval_data_folder", type=str, default="replay_data")
    parser.add_argument("--record_game", action="store_true")
    parser.add_argument("--subj", type=str, default=None)
    parser.add_argument("--run_model", action="store_true")
    parser.add_argument("--eval_data", action="store_true")
    args = parser.parse_args()
    assert sum([args.record_game, args.run_model, args.eval_data]) == 1

    if args.record_game:
        record_game(args)
    elif args.run_model:
        run_model(args)
    else:
        evaluate_data(args)

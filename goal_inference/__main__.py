from goal_inference.game import Game
from goal_inference.replay import Replay
from goal_inference.world import all_worlds
import argparse
import pathlib
import numpy as np
import os
from tqdm import tqdm  # type: ignore[import-untyped]
import gc
from itertools import product
import logging
from joblib import Parallel, delayed, parallel_backend  # type: ignore[import-untyped]


def hyperparam_search(num_alphas, num_n_turns, num_p_actions, num_p_goals):
    alphas = np.logspace(0, 5, num=num_alphas, base=4)
    n_turns = np.arange(1, num_n_turns + 1)
    p_actions = np.linspace(0.01, 0.65, num=num_p_actions)
    p_goals = np.linspace(0.01, 0.65, num=num_p_goals)
    logging.info(f"Testing alphas in {list(alphas)}")
    logging.info(f"Testing n_turns in {n_turns}")
    logging.info(f"Testing p_actions in {p_actions}")
    logging.info(f"Testing p_goals in {p_goals}")
    for alpha in alphas:
        for n in n_turns:
            yield alpha, ("turn", n)
        for p in p_actions:
            yield alpha, ("action", p)
        for p in p_goals:
            yield alpha, ("goal", p)


def run_generator(num_alphas, num_n_turns, num_p_actions, num_p_goals):
    logging.info(f"Starting a run sweep over {len(all_worlds)} worlds")
    for (idx, world), hyperparams in tqdm(
        product(
            enumerate(all_worlds),
            hyperparam_search(num_alphas, num_n_turns, num_p_actions, num_p_goals),
        ),
        total=(
            len(all_worlds) * num_alphas * (num_n_turns + num_p_actions + num_p_goals)
        ),
    ):
        yield hyperparams, (idx, world)


def replay_generator(folder, num_alphas, num_n_turns, num_p_actions, num_p_goals):
    files = list(folder.iterdir())
    logging.info(f"Starting a replay sweep over {len(files)} recordings")
    for hyperparams, human_csv in tqdm(
        product(
            hyperparam_search(num_alphas, num_n_turns, num_p_actions, num_p_goals),
            files,
        ),
        total=(len(files) * num_alphas * (num_n_turns + num_p_actions + num_p_goals)),
    ):
        idx = int(human_csv.stem.split("_")[1])
        world = all_worlds[idx]
        yield hyperparams, (world, human_csv)


def get_settings(
    mode, folder=None, num_alphas=6, num_n_turns=5, num_p_actions=5, num_p_goals=5
):
    assert mode == "run" or mode == "replay"
    if mode == "replay":
        assert folder is not None
        return replay_generator(
            folder, num_alphas, num_n_turns, num_p_actions, num_p_goals
        )
    else:
        return run_generator(num_alphas, num_n_turns, num_p_actions, num_p_goals)


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
    logging.info(f"Subject: {args.subj}, Worlds explored: {len(played_worlds)}")
    assert len(played_worlds) < len(all_worlds), "You have already played all worlds!"
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

    def play_game(settings):
        (alpha, update_criteria), (idx, world) = settings
        game = Game(
            world=world,
            human_player=False,
            record=True,
            output_folder=folder,
            csv_name=f"update={update_criteria}_alpha={alpha}_{idx}.csv",
            alpha=alpha,
            update_criteria=update_criteria,
            gui=False,
        )
        game.play()
        del game
        gc.collect()

    with parallel_backend("loky", n_jobs=-2):
        Parallel()(delayed(play_game)(settings) for settings in get_settings("run"))


def evaluate_data(args):
    output_folder = pathlib.Path(args.eval_data_folder)
    input_folder = pathlib.Path(args.record_game_folder)
    assert os.path.exists(input_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    def replay_game(settings):
        (alpha, update_criteria), (world, human_csv) = settings
        replay = Replay(
            world=world,
            human_csv=human_csv,
            alpha=alpha,
            update_criteria=update_criteria,
        )
        data = replay.replay()
        output_file = human_csv.stem + f"_update={update_criteria}_alpha={alpha}.csv"
        data.to_csv(output_folder / output_file, index=False)
        del replay
        del data
        gc.collect()

    with parallel_backend("loky", n_jobs=-2):
        Parallel()(
            delayed(replay_game)(settings)
            for settings in get_settings("replay", input_folder)
        )


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

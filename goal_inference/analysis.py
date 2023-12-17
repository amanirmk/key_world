import pandas as pd  # type: ignore[import-untyped]
import argparse
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore[import-untyped]
import numpy as np


def extract_data(input_folder, mode):
    dfs = []
    for file in pathlib.Path(input_folder).iterdir():
        df = pd.read_csv(file)
        if mode == "human":
            df["subj"], idx_str = file.stem.split("_")
            df["world"] = int(idx_str)
        elif mode == "model":
            update_criteria_str, alpha_str, idx_str = file.stem.split("_")
            df["world"] = int(idx_str)
            df["alpha"] = float(alpha_str.split("=")[1])
            df["update_criteria"] = [eval(update_criteria_str.split("=")[1])] * len(
                df.index
            )
        elif mode == "replay":
            df["subj"], idx, update_criteria_str, alpha_str = file.stem.split("_")
            df["world"] = int(idx)
            df["alpha"] = float(alpha_str.split("=")[1])
            df["update_criteria"] = [eval(update_criteria_str.split("=")[1])] * len(
                df.index
            )
        else:
            return NotImplemented
        dfs.append(df)
    return pd.concat(dfs)


def moves_until_pos(move_list, pos_str):
    i = len(move_list) + 1
    while move_list[i - 2] == pos_str:
        i -= 1
        if i - 2 < 0:
            return pd.NA
    return i


def get_performance(data, groupby, avg_over=[]):
    knower_moves = data.groupby(groupby + avg_over, as_index=False)["knower_pos"].apply(
        list
    )
    watcher_moves = data.groupby(groupby + avg_over, as_index=False)[
        "watcher_pos"
    ].apply(list)
    watcher_moves["num_rounds"] = watcher_moves["watcher_pos"].apply(len)
    assert all(watcher_moves["num_rounds"] <= 100)
    watcher_moves["solved"] = watcher_moves["num_rounds"] < 100
    watcher_moves["watcher_door_speed"] = watcher_moves["watcher_pos"].apply(
        lambda moves: moves_until_pos(moves, "(9, 10)")
    )
    watcher_moves["knower_door_speed"] = knower_moves["knower_pos"].apply(
        lambda moves: moves_until_pos(moves, "(9, 9)")
    )
    watcher_moves["speed_ratio"] = (
        watcher_moves["knower_door_speed"] / watcher_moves["watcher_door_speed"]
    )
    watcher_moves["speed_diff"] = (
        watcher_moves["knower_door_speed"] - watcher_moves["watcher_door_speed"]
    )
    summary = (
        watcher_moves[
            groupby
            + [
                "num_rounds",
                "solved",
                "watcher_door_speed",
                "knower_door_speed",
                "speed_ratio",
                "speed_diff",
            ]
        ]
        .groupby(groupby, as_index=False)
        .mean()
    )  # ignores NA
    return summary.sort_index()


def get_comparison(data, groupby, avg_over=[]):
    g = data.groupby(groupby + avg_over, as_index=False)
    df = g["log_likelihood"].apply(list)
    df["action_surprisal"] = g["action_surprisal"].apply(list)["action_surprisal"]
    df["goal_surprisal"] = g["goal_surprisal"].apply(list)["goal_surprisal"]
    df["reaction_time"] = g["reaction_time"].apply(list)["reaction_time"]
    df["avg_log_likelihood"] = df["log_likelihood"].apply(np.mean)
    return NotImplemented

def make_heatmaps(
    summary, groupby, metric_cols=["speed_ratio", "speed_diff", "solved"]
):
    g = summary.groupby(groupby)
    for col in metric_cols:
        m = g[col].mean().unstack(level=0)
        sns.heatmap(m, annot=True)
        plt.title(col)
        plt.show()
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_game_folder", type=str, default="human_data")
    parser.add_argument("--run_model_folder", type=str, default="model_data")
    parser.add_argument("--replay_data_folder", type=str, default="replay_data")
    parser.add_argument("--human_data", action="store_true")
    parser.add_argument("--model_data", action="store_true")
    parser.add_argument("--replay_data", action="store_true")
    args = parser.parse_args()

    if args.human_data:
        data = extract_data(args.record_game_folder, mode="human")
        performance_summary = get_performance(
            data, groupby=["world"], avg_over=["subj"]
        )
        make_heatmaps(performance_summary, groupby=["world", "solved"])
    if args.model_data:
        data = extract_data(args.run_model_folder, mode="model")
        performance_summary = get_performance(
            data, groupby=["world", "alpha", "update_criteria"]
        )
        make_heatmaps(performance_summary, groupby=["alpha", "update_criteria"])
    if args.replay_data:
        data = extract_data(args.replay_data_folder, mode="replay")
        comparison_summary = get_comparison(
            data, groupby=["world", "alpha", "update_criteria"], avg_over=["subj"]
        )
        make_heatmaps(comparison_summary, groupby=["alpha", "update_criteria"])

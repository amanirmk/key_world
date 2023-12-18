import pandas as pd  # type: ignore[import-untyped]
import argparse
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore[import-untyped]
import numpy as np
from scipy.stats import pearsonr  # type: ignore[import-untyped]
from goal_inference.game import Game


def format_col(col):
    if col == "alpha":
        return r"$\alpha$"
    return col.replace("_", " ").replace("avg", "average").title()


def format_update(update_criteria):
    if isinstance(update_criteria, str):
        update_criteria = eval(update_criteria)
    kind, val = update_criteria
    if kind in ["action", "goal"]:
        c = "p"
    else:
        c = "n"
    return rf"{kind}: ${c}={val}$"


def format_alpha(alpha):
    return rf"$\alpha={int(alpha)}$"


def format_settings(update_criteria, alpha):
    return rf"{format_update(update_criteria)}, {format_alpha(alpha)}"


def extract_data(input_folder, mode):
    dfs = []
    for file in pathlib.Path(input_folder).iterdir():
        df = pd.read_csv(file)
        if mode == "human":
            df["subj"], idx_str = file.stem.split("_")
            df["world"] = int(idx_str)
        elif mode == "model":
            update_criteria_str, alpha_str, idx_str = file.stem.split("_")
            df["beliefs"] = df["beliefs"].apply(eval)
            df["world"] = int(idx_str)
            df["alpha"] = int(float(alpha_str.split("=")[1]))
            df["update_criteria"] = [eval(update_criteria_str.split("=")[1])] * len(
                df.index
            )
        elif mode == "replay":
            df["subj"], idx, update_criteria_str, alpha_str = file.stem.split("_")
            df["world"] = int(idx)
            df["alpha"] = int(float(alpha_str.split("=")[1]))
            df["update_criteria"] = [eval(update_criteria_str.split("=")[1])] * len(
                df.index
            )
        else:
            return NotImplemented
        dfs.append(df)
    return pd.concat(dfs)


def moves_until_pos(move_list, pos_str, is_watcher):
    if is_watcher and len(move_list) == 100:
        return 100
    i = len(move_list) + 1
    while move_list[i - 2] == pos_str:
        i -= 1
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
        lambda moves: moves_until_pos(moves, "(9, 10)", is_watcher=True)
    )
    watcher_moves["knower_door_speed"] = knower_moves["knower_pos"].apply(
        lambda moves: moves_until_pos(moves, "(9, 9)", is_watcher=False)
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
    )
    return summary.sort_index()


def get_comparison(data, groupby, avg_over=[]):
    g = data.groupby(groupby + avg_over, as_index=False)
    df = g["log_likelihood"].apply(list)
    df["action_surprisal"] = g["action_surprisal"].apply(list)["action_surprisal"]
    df["goal_surprisal"] = g["goal_surprisal"].apply(list)["goal_surprisal"]
    df["reaction_time"] = g["reaction_time"].apply(list)["reaction_time"]
    df["avg_log_likelihood"] = df["log_likelihood"].apply(np.mean)
    summary = (
        df[
            groupby
            + [
                "avg_log_likelihood",
            ]
        ]
        .groupby(groupby)
        .mean()
    )
    summary["goal_correlation_r"] = 0.0
    summary["goal_correlation_p"] = 0.0
    summary["action_correlation_r"] = 0.0
    summary["action_correlation_p"] = 0.0
    for idx, group in df.groupby(groupby):
        all_goal_surprisals = sum(
            (seq[1:] for seq in group["goal_surprisal"].values), []
        )
        all_action_surprisals = sum(
            (seq[1:] for seq in group["action_surprisal"].values), []
        )
        all_reaction_times = sum((seq[1:] for seq in group["reaction_time"].values), [])
        (
            summary["goal_correlation_r"][idx],
            summary["goal_correlation_p"][idx],
        ) = pearsonr(all_goal_surprisals, all_reaction_times)
        (
            summary["action_correlation_r"][idx],
            summary["action_correlation_p"][idx],
        ) = pearsonr(all_action_surprisals, all_reaction_times)
    return summary.reset_index()


def make_heatmaps(output_folder, summary, groupby, metric_cols):
    assert len(groupby) == 2
    assert groupby[0] != "dummy"
    summary["dummy"] = 1
    g = summary.groupby(groupby)
    plt.rc("text", usetex=True)
    for col in metric_cols:
        m = g[col].mean().unstack(level=0)
        if groupby[1] == "dummy":
            plt.figure(figsize=(16, 1))
        else:
            plt.figure(figsize=(16, 4))
        sns.heatmap(m, annot=True)
        if groupby[0] == "update_criteria":
            for y in [5, 10]:
                plt.axvline(y, color="white", linewidth=5.0)
            plt.gca().set_xticklabels(
                [
                    format_update(tick_label.get_text())
                    for tick_label in plt.gca().get_xticklabels()
                ]
            )
        plt.title(format_col(col))
        plt.xlabel(format_col(groupby[0]))
        plt.ylabel(format_col(groupby[1]) if groupby[1] != "dummy" else None)
        if groupby[1] == "dummy":
            plt.yticks([])
        else:
            plt.yticks(rotation=0)
        plt.savefig(
            output_folder / f"{'-'.join([col] + groupby)}.pdf", bbox_inches="tight"
        )
        plt.close()


def make_belief_plots(output_folder, model_data):
    plt.style.use("ggplot")
    plt.rc("text", usetex=True)
    belief_data = model_data.groupby(
        ["world", "update_criteria", "alpha"], as_index=False
    )["beliefs"].apply(list)
    for world in belief_data["world"].unique():
        world_data = belief_data[belief_data["world"] == world]
        for (update_criteria, alpha), beliefs in world_data.groupby(
            ["update_criteria", "alpha"]
        )["beliefs"]:
            goals = list(beliefs.iloc[0][0].keys())
            goal_data = [[b[g] for b in beliefs.iloc[0]] for g in goals]
            plt.figure(figsize=(3, 1.35))
            for g, g_data in zip(goals, goal_data):
                plt.plot(
                    g_data, color=(np.array(Game.key_colors(g)) / 255), linewidth=3
                )
            plt.ylim((0, 1))
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                output_folder / f"{world}-{update_criteria}-{alpha}.pdf",
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()


def analyze_data(
    record_game_folder=None, run_model_folder=None, replay_data_folder=None
):
    if record_game_folder:
        data = extract_data(record_game_folder, mode="human")
        performance_summary = get_performance(
            data, groupby=["world"], avg_over=["subj"]
        )
        output_folder = pathlib.Path(record_game_folder + "_outputs")
        output_folder.mkdir(parents=True, exist_ok=True)
        make_heatmaps(
            output_folder,
            performance_summary,
            groupby=["world", "dummy"],
            metric_cols=["speed_ratio", "speed_diff", "solved"],
        )

    if run_model_folder:
        data = extract_data(run_model_folder, mode="model")
        performance_summary = get_performance(
            data, groupby=["world", "alpha", "update_criteria"]
        )
        output_folder = pathlib.Path(run_model_folder + "_outputs")
        belief_plot_folder = output_folder / "belief_plots"
        belief_plot_folder.mkdir(parents=True, exist_ok=True)
        make_heatmaps(
            output_folder,
            performance_summary,
            groupby=["update_criteria", "alpha"],
            metric_cols=["speed_ratio", "speed_diff", "solved"],
        )
        make_heatmaps(
            output_folder,
            performance_summary,
            groupby=["world", "dummy"],
            metric_cols=["speed_ratio", "speed_diff", "solved"],
        )
        make_belief_plots(belief_plot_folder, data)

    if replay_data_folder:
        data = extract_data(replay_data_folder, mode="replay")
        comparison_summary = get_comparison(
            data, groupby=["alpha", "update_criteria"], avg_over=["subj", "world"]
        )
        output_folder = pathlib.Path(replay_data_folder + "_outputs")
        output_folder.mkdir(parents=True, exist_ok=True)
        make_heatmaps(
            output_folder,
            comparison_summary,
            groupby=["update_criteria", "alpha"],
            metric_cols=[
                "avg_log_likelihood",
                "goal_correlation_r",
                "action_correlation_r",
                "goal_correlation_p",
                "action_correlation_p",
            ],
        )

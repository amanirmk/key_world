from goal_inference.game import Game
from goal_inference.replay import Replay
from goal_inference.world import example_world

if __name__ == "__main__":
    # Game(
    #     world=example_world,
    #     human_player=True,
    #     record=True,
    #     output_folder="./outputs",
    #     csv_name="log.csv",
    #     # don't matter
    #     alpha=20,
    #     update_criteria=("turn", 1),
    # ).play()

    df, summary = Replay(
        world=example_world,
        human_csv="./outputs/log.csv",
        # matter
        alpha=20,
        update_criteria=("turn", 1),
    ).replay()

    print(summary)
    df.to_csv("./outputs/replay.csv")

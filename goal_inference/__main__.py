from goal_inference.game import Game
from goal_inference.world import example_world

if __name__ == "__main__":
    Game(
        world=example_world,
        human_player=False,
        alpha=10,
        update_criteria=("turn", 3),
        record=False,
        output_folder="./outputs",
        csv_name="log.csv",
    )

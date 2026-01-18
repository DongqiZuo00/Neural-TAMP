from scripts.data_collection import run_rollouts, _print_summary


def main() -> None:
    result = run_rollouts(
        num_scenes=1,
        steps_per_scene=50,
        output_dir="datasets/phase1/",
        seed=0,
        policy="random",
        save_mode="jsonl",
    )
    _print_summary(result)


if __name__ == "__main__":
    main()

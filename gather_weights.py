from pathlib import Path

weights_paths = {}

for path in Path("./code/results/").iterdir():
    dataset = path.name
    weights_paths[dataset] = {}

    for seed_dir in (path / "self_supervised").iterdir():
        seed = seed_dir.name
        if seed not in weights_paths[dataset]:
            weights_paths[dataset][seed] = []
        for weight_path in seed_dir.glob("**/epoch_500-cifar.pt"):
            weights_paths[dataset][seed].append(weight_path.absolute())


for dataset in weights_paths.keys():
    for seed, paths in weights_paths[dataset].items():
        target_path = Path("./scripts") / dataset / "eval" / seed / "all_weights.txt"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w") as f:
            for path in paths:
                f.write(f"{str(path)}\n")

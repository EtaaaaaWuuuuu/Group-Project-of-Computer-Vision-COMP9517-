import random
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _get_paths_and_labels(root_dir):
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root_dir}"
        )

    paths = []
    labels = []

    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue

        image_paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTS
        )

        for image_path in image_paths:
            paths.append(image_path)
            labels.append(folder.name)

    return paths, labels


def build_subset(cfg, train_select_root, test_select_root):
    rng = random.Random(cfg.seed)

    full_paths, full_labels = _get_paths_and_labels(
        train_select_root
    )

    test_paths_all, test_labels_all = _get_paths_and_labels(
        test_select_root
    )

    paths_by_species = defaultdict(list)

    for path, label in zip(full_paths, full_labels):
        paths_by_species[label].append(path)

    test_by_species = defaultdict(list)

    for path, label in zip(test_paths_all, test_labels_all):
        test_by_species[label].append(path)

    eligible_species = []

    for species, paths in paths_by_species.items():
        if species not in test_by_species:
            continue

        if (
            len(paths)
            >= cfg.n_train_per_class + cfg.n_val_per_class
            and len(test_by_species[species])
            >= cfg.n_test_per_class
        ):
            eligible_species.append(species)

    if len(eligible_species) < cfg.n_classes:
        raise ValueError(
            f"Requested {cfg.n_classes} classes, but only "
            f"{len(eligible_species)} classes contain enough images."
        )

    if len(eligible_species) == cfg.n_classes:
        selected_species = eligible_species
    else:
        selected_species = sorted(
            rng.sample(
                eligible_species,
                cfg.n_classes,
            )
        )

    class_names = list(selected_species)

    label_mapping = {
        species: label
        for label, species in enumerate(class_names)
    }

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for species in selected_species:
        label = label_mapping[species]

        paths = sorted(paths_by_species[species])
        rng.shuffle(paths)

        train_paths = paths[:cfg.n_train_per_class]

        val_start = cfg.n_train_per_class
        val_end = val_start + cfg.n_val_per_class

        val_paths = paths[val_start:val_end]

        test_paths = sorted(
            test_by_species[species]
        )[:cfg.n_test_per_class]

        splits["train"].extend(
            (str(path), label)
            for path in train_paths
        )

        splits["val"].extend(
            (str(path), label)
            for path in val_paths
        )

        splits["test"].extend(
            (str(path), label)
            for path in test_paths
        )

    validate_splits(
        splits,
        class_names,
        cfg,
    )

    return splits, class_names


def validate_splits(splits, class_names, cfg):
    expected_train = (
        cfg.n_classes
        * cfg.n_train_per_class
    )

    expected_val = (
        cfg.n_classes
        * cfg.n_val_per_class
    )

    expected_test = (
        cfg.n_classes
        * cfg.n_test_per_class
    )

    if len(class_names) != cfg.n_classes:
        raise AssertionError(
            f"Expected {cfg.n_classes} classes, "
            f"but found {len(class_names)}."
        )

    if len(splits["train"]) != expected_train:
        raise AssertionError(
            f"Expected {expected_train} training images, "
            f"but found {len(splits['train'])}."
        )

    if len(splits["val"]) != expected_val:
        raise AssertionError(
            f"Expected {expected_val} validation images, "
            f"but found {len(splits['val'])}."
        )

    if len(splits["test"]) != expected_test:
        raise AssertionError(
            f"Expected {expected_test} test images, "
            f"but found {len(splits['test'])}."
        )

    train_paths = {
        path for path, _ in splits["train"]
    }

    val_paths = {
        path for path, _ in splits["val"]
    }

    test_paths = {
        path for path, _ in splits["test"]
    }

    if not train_paths.isdisjoint(val_paths):
        raise AssertionError(
            "Training and validation sets overlap."
        )

    if not train_paths.isdisjoint(test_paths):
        raise AssertionError(
            "Training and test sets overlap."
        )

    if not val_paths.isdisjoint(test_paths):
        raise AssertionError(
            "Validation and test sets overlap."
        )

    print("Dataset split successfully validated.")
    print(f"Classes:    {len(class_names):,}")
    print(f"Training:   {len(train_paths):,}")
    print(f"Validation: {len(val_paths):,}")
    print(f"Test:       {len(test_paths):,}")
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import json


@dataclass
class Config:

    exp_name: str = "baseline"

    train_select_root: str = "../train_select"
    test_select_root: str = "../test_select"

    results_dir: str = "results"

    n_classes: int = 1000
    n_train_per_class: int = 40
    n_val_per_class: int = 10
    n_test_per_class: int = 10
    seed: int = 20269517

    image_size: int = 256

    sift_mode: str = "sparse"
    dense_step: int = 8
    dense_scales: tuple = (8, 16)
    n_sift_features: int = 0
    contrast_threshold: float = 0.04
    max_desc_per_img: int = 500
    rootsift: bool = True

    codebook_size: int = 1024
    kmeans_sample_size: int = 500_000
    kmeans_batch_size: int = 10_000
    kmeans_max_iter: int = 100
    kmeans_n_init: int = 3

    use_spm: bool = False  # True or False
    clf: str = "linear_svm"

    encoding: str = "hard"  # "hard" or "soft"
    soft_k: int = 5
    soft_sigma: float = 0.0
    use_tfidf: bool = True  # True or False
    norm: str = "l2"

    C: float = 1.0
    max_iter: int = 3000
    rf_n_estimators: int = 200
    rf_max_depth: int = 20

    n_jobs: int = -1
    verbose: bool = False

    @property
    def feature_dim(self) -> int:
        return self.codebook_size * (5 if self.use_spm else 1)

    def variant(self, **kwargs) -> "Config":
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def exp_dir(self) -> Path:
        p = Path(self.results_dir) / self.exp_name
        p.mkdir(parents=True, exist_ok=True)
        return p


CFG = Config()

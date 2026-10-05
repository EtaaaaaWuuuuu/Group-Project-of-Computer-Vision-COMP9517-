import numpy as np


def _soft_sigma(sample_descriptors, kmeans):
 
    dists = kmeans.transform(sample_descriptors)
    nearest = np.sort(dists, axis=1)[:, 0]
    sigma = np.median(nearest) + 1e-6
    return sigma


def _histogram_hard(descriptors, kmeans):
    labels = kmeans.predict(descriptors)
    hist = np.bincount(labels, minlength=kmeans.n_clusters).astype(np.float64)
    return hist


def _histogram_soft(descriptors, kmeans, sigma, k):
    dists = kmeans.transform(descriptors)
    k = min(k, dists.shape[1])
    nearest_idx = np.argpartition(dists, k-1, axis=1)[:, :k]

    hist = np.zeros(kmeans.n_clusters, dtype=np.float64)
    for i in range(dists.shape[0]):
        idx = nearest_idx[i]
        d = dists[i, idx]
        w = np.exp(-(d ** 2) / (2 * sigma ** 2))
        w_sum = w.sum()
        if w_sum > 0:
            hist[idx] += w / w_sum
    return hist


def _normalize(hist, norm):
    if norm == "l1":
        s = np.abs(hist).sum()
    else:
        s = np.sqrt((hist ** 2).sum())
    return hist / s if s > 1e-12 else hist


def _cell_index(x, y, w, h, grid):
    col = min(grid - 1, int(x / w * grid))
    row = min(grid - 1, int(y / h * grid))
    return row * grid + col


def encode_image(coords, hw, descriptors, kmeans, cfg, sigma=None):
    if descriptors.shape[0] == 0:
        n_cells = 1 if not cfg.use_spm else 5
        return np.zeros(kmeans.n_clusters * n_cells, dtype=np.float64)

    def _hist(desc):
        if cfg.encoding == "hard":
            return _histogram_hard(desc, kmeans)
        elif cfg.encoding == "soft":
            return _histogram_soft(desc, kmeans, sigma, cfg.soft_k)
        else:
            raise ValueError(f"Unknown encoding: {cfg.encoding}")

    if not cfg.use_spm:
        hist = _hist(descriptors)
        return _normalize(hist, cfg.norm)

    h, w = hw
    level0 = _normalize(_hist(descriptors), cfg.norm) * 0.5

    grid = 2
    cell_ids = np.array([_cell_index(x, y, w, h, grid) for x, y in coords])
    level1_parts = []
    for c in range(grid * grid):
        mask = cell_ids == c
        if mask.sum() == 0:
            level1_parts.append(np.zeros(kmeans.n_clusters))
        else:
            level1_parts.append(_normalize(_hist(descriptors[mask]), cfg.norm))
    level1 = np.concatenate(level1_parts) * (0.5 / (grid * grid))

    feature = np.concatenate(
        [level0, level1]
    )

    return _normalize(
        feature,
        cfg.norm,
    )


def compute_idf(hard_counts_matrix):
    n_images = hard_counts_matrix.shape[0]
    df = (hard_counts_matrix > 0).sum(axis=0)
    idf = np.log((n_images + 1) / (df + 1)) + 1.0
    return idf


def encode_from_cache(
    cache,
    kmeans,
    cfg,
    idf=None,
    sigma=None,
):
    vectors = []
    labels = []

    for coords, hw, desc, label in cache:
        vector = encode_image(
            coords,
            hw,
            desc,
            kmeans,
            cfg,
            sigma=sigma,
        )

        vectors.append(vector)
        labels.append(label)

    X = np.stack(
        vectors,
        axis=0,
    ).astype(np.float32)

    y = np.asarray(
        labels,
        dtype=np.int32,
    )

    if idf is not None:
        n_cells = X.shape[1] // len(idf)

        X = X * np.tile(
            idf,
            n_cells,
        )

        norms = np.linalg.norm(
            X,
            axis=1,
            keepdims=True,
        )

        X = X / np.maximum(
            norms,
            1e-12,
        )

    return X.astype(np.float32), y
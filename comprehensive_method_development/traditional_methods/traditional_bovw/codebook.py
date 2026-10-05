import numpy as np
from sklearn.cluster import MiniBatchKMeans


def build_pool_from_cache(train_cache, cfg):

    all_desc = [desc for _, _, desc, _ in train_cache if desc.shape[0] > 0]
    all_desc = np.concatenate(all_desc, axis=0) if all_desc else np.empty((0, 128), np.float32)

    if all_desc.shape[0] > cfg.kmeans_sample_size:
        rng = np.random.RandomState(cfg.seed)
        idx = rng.choice(all_desc.shape[0], size=cfg.kmeans_sample_size, replace=False)
        all_desc = all_desc[idx]

    return all_desc


def build_codebook(descriptors, cfg):
    kmeans = MiniBatchKMeans(
        n_clusters=cfg.codebook_size,
        batch_size=cfg.kmeans_batch_size,
        random_state=cfg.seed,
        n_init=cfg.kmeans_n_init,
        max_iter=cfg.kmeans_max_iter,
        verbose=int(cfg.verbose),
    )
    kmeans.fit(descriptors)
    return kmeans
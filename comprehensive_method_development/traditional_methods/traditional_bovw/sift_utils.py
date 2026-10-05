import cv2
import numpy as np


def resize_keep_aspect(img, target_size):

    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_root_sift(descriptors, eps=1e-7):

    if descriptors is None or len(descriptors) == 0:
        return descriptors
    descriptors = descriptors.astype(np.float32)
    descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
    descriptors = np.sqrt(descriptors)
    return descriptors


def _make_dense_keypoints(h, w, step, scales):
    keypoints = []
    for size in scales:
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                keypoints.append(cv2.KeyPoint(float(x), float(y), float(size)))
    return keypoints


def extract_descriptors(image_path, cfg, sift):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = resize_keep_aspect(img, cfg.image_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    if cfg.sift_mode == "sparse":
        keypoints, descriptors = sift.detectAndCompute(gray, None)
    elif cfg.sift_mode == "dense":
        keypoints = _make_dense_keypoints(h, w, cfg.dense_step, cfg.dense_scales)
        keypoints, descriptors = sift.compute(gray, keypoints)
    else:
        raise ValueError(f"Unknown sift_mode: {cfg.sift_mode}")

    if descriptors is None:
        return np.empty((0, 2), dtype=np.float32), (h, w), np.empty((0, 128), dtype=np.float32)

    if cfg.rootsift:
        descriptors = to_root_sift(descriptors)

    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    n = descriptors.shape[0]
    if n > cfg.max_desc_per_img:
        if cfg.sift_mode == "sparse":
            responses = np.array([kp.response for kp in keypoints])
            keep = np.argsort(-responses)[: cfg.max_desc_per_img]
        else:
            rng = np.random.RandomState(cfg.seed)
            keep = rng.choice(n, size=cfg.max_desc_per_img, replace=False)
        descriptors = descriptors[keep]
        coords = coords[keep]

    return coords, (h, w), descriptors


def get_sift_detector(cfg):
    
    return cv2.SIFT_create(
        nfeatures=cfg.n_sift_features,
        contrastThreshold=cfg.contrast_threshold,
    )
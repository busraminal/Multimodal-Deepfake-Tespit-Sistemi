import numpy as np

def mouth_embed_dummy(frames_dir, dim=384):
    rng = np.random.default_rng(42)
    v = rng.normal(size=(dim,))
    v = v / np.linalg.norm(v)
    return v.astype(np.float32)

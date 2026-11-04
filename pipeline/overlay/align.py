"""Time-align two reps: contact frame pinned to contact frame, DTW before and after it."""
import numpy as np


def frame_cost(a, ca, b, cb):
    """Pairwise cost between every frame of a (n, J, 2) and b (m, J, 2).

    Mean joint distance in normalized units, weighted by the lower confidence of the pair.
    """
    d = np.linalg.norm(a[:, None] - b[None, :], axis=-1)          # (n, m, J)
    w = np.minimum(ca[:, None], cb[None, :])                     # (n, m, J)
    wsum = w.sum(-1)
    return np.where(wsum > 0, (d * w).sum(-1) / np.maximum(wsum, 1e-9), d.mean(-1))


def dtw(cost):
    """Classic DTW over a cost matrix; path runs from (0, 0) to (n-1, m-1)."""
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]


def align(new, old_norm, old_conf, old_contact):
    """For every frame of `new`, the list of matched frame indices in the old rep.

    The two segments [start, contact] and [contact, end] are warped separately, so the
    contact frames always line up exactly.
    """
    cn, co = new.contact, old_contact
    matches = [[] for _ in range(len(new.frames))]
    for (a0, a1), (b0, b1) in (((0, cn + 1), (0, co + 1)),
                               ((cn, len(new.frames)), (co, len(old_norm)))):
        cost = frame_cost(new.norm[a0:a1], new.conf[a0:a1], old_norm[b0:b1], old_conf[b0:b1])
        for i, j in dtw(cost):
            matches[a0 + i].append(b0 + j)
    matches[cn] = [co]
    return [sorted(set(m)) for m in matches]


def ghost_sequence(new, old_norm, old_conf, old_contact):
    """Ghost pose per new frame: mean of the old frames DTW matched to it.

    Returns (ghost (T, J, 2), ghost_conf (T, J), old_index per new frame).
    """
    matches = align(new, old_norm, old_conf, old_contact)
    ghost = np.stack([old_norm[m].mean(0) for m in matches])
    gconf = np.stack([old_conf[m].min(0) for m in matches])
    old_idx = [m[len(m) // 2] for m in matches]
    return ghost, gconf, old_idx

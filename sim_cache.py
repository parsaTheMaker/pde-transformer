import glob
import os
import tempfile
from multiprocessing import get_context

import numpy as np


def load_npz_array(path):
    with np.load(path) as data:
        return data["arr_0"].astype(np.float32)


def load_packed_array(path, mmap_mode="r"):
    return np.load(path, mmap_mode=mmap_mode)


def discover_simulations(root_dir):
    sim_dirs = sorted(
        path for path in glob.glob(os.path.join(root_dir, "sim_*"))
        if os.path.isdir(path)
    )
    sim_infos = []
    for sim_dir in sim_dirs:
        vel_files = sorted(glob.glob(os.path.join(sim_dir, "velocity_*.npz")))
        pre_files = sorted(glob.glob(os.path.join(sim_dir, "pressure_*.npz")))
        if not vel_files or not pre_files:
            continue
        if len(vel_files) != len(pre_files):
            raise RuntimeError(
                f"Frame count mismatch in {sim_dir}: vel={len(vel_files)} pre={len(pre_files)}"
            )
        sim_infos.append(
            {
                "dir": sim_dir,
                "vel": vel_files,
                "pre": pre_files,
                "mask_path": os.path.join(sim_dir, "obstacle_mask.npz"),
                "n_frames": len(vel_files),
            }
        )

    if not sim_infos:
        raise RuntimeError(f"No valid simulation folders found under {root_dir}")

    return sim_infos


def _get_sim_cache_paths(sim_dir, states_filename, mask_filename):
    return (
        os.path.join(sim_dir, states_filename),
        os.path.join(sim_dir, mask_filename),
    )


def _infer_sim_shapes(sim_info):
    vel0 = load_npz_array(sim_info["vel"][0])
    pre0 = load_npz_array(sim_info["pre"][0])
    if vel0.ndim != 3 or vel0.shape[0] != 2:
        raise RuntimeError(f"Unexpected velocity shape in {sim_info['dir']}: {vel0.shape}")
    if pre0.ndim != 3 or pre0.shape[0] != 1:
        raise RuntimeError(f"Unexpected pressure shape in {sim_info['dir']}: {pre0.shape}")
    if vel0.shape[1:] != pre0.shape[1:]:
        raise RuntimeError(
            f"Velocity/pressure spatial shape mismatch in {sim_info['dir']}: {vel0.shape} vs {pre0.shape}"
        )
    return (sim_info["n_frames"], 3, vel0.shape[1], vel0.shape[2]), (1, vel0.shape[1], vel0.shape[2])


def _load_source_mask(sim_info, expected_mask_shape):
    if os.path.exists(sim_info["mask_path"]):
        mask = load_npz_array(sim_info["mask_path"])
    else:
        mask = np.ones(expected_mask_shape[1:], dtype=np.float32)

    if mask.ndim == 2:
        mask = mask[None, ...]
    elif mask.ndim != 3:
        raise RuntimeError(f"Unexpected obstacle mask shape in {sim_info['dir']}: {mask.shape}")

    if tuple(mask.shape) != tuple(expected_mask_shape):
        raise RuntimeError(
            f"Obstacle mask shape mismatch in {sim_info['dir']}: expected {expected_mask_shape}, got {mask.shape}"
        )

    return np.ascontiguousarray(mask.astype(np.float32, copy=False))


def _validate_sim_cache(sim_info):
    states_path = sim_info["states_path"]
    packed_mask_path = sim_info["packed_mask_path"]
    expected_states_shape = tuple(sim_info["states_shape"])
    expected_mask_shape = tuple(sim_info["mask_shape"])

    if not (os.path.exists(states_path) and os.path.exists(packed_mask_path)):
        return False

    try:
        states = load_packed_array(states_path)
        mask = load_packed_array(packed_mask_path)
    except (OSError, ValueError):
        return False

    return (
        states.dtype == np.float32
        and mask.dtype == np.float32
        and tuple(states.shape) == expected_states_shape
        and tuple(mask.shape) == expected_mask_shape
    )


def _write_array_atomic(path, array):
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".tmp_", suffix=".npy")
    os.close(fd)
    try:
        np.save(tmp_path, np.ascontiguousarray(array, dtype=np.float32))
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _build_sim_cache(sim_info):
    states_path = sim_info["states_path"]
    packed_mask_path = sim_info["packed_mask_path"]
    states_shape = tuple(sim_info["states_shape"])
    mask_shape = tuple(sim_info["mask_shape"])
    sim_name = os.path.basename(sim_info["dir"])

    print(f"Packing {sim_name} -> {os.path.basename(states_path)}")

    fd, tmp_states_path = tempfile.mkstemp(dir=sim_info["dir"], prefix=".tmp_states_", suffix=".npy")
    os.close(fd)

    try:
        states_mm = np.lib.format.open_memmap(
            tmp_states_path,
            mode="w+",
            dtype=np.float32,
            shape=states_shape,
        )
        for frame_idx, (vel_path, pre_path) in enumerate(zip(sim_info["vel"], sim_info["pre"])):
            vel = load_npz_array(vel_path)
            pre = load_npz_array(pre_path)
            states_mm[frame_idx, :2] = vel
            states_mm[frame_idx, 2:3] = pre
        states_mm.flush()
        del states_mm
        os.replace(tmp_states_path, states_path)
    finally:
        if os.path.exists(tmp_states_path):
            os.unlink(tmp_states_path)

    mask = _load_source_mask(sim_info, mask_shape)
    _write_array_atomic(packed_mask_path, mask)


def prepare_sim_cache_info(sim_info, states_filename, mask_filename):
    states_shape, mask_shape = _infer_sim_shapes(sim_info)
    states_path, packed_mask_path = _get_sim_cache_paths(sim_info["dir"], states_filename, mask_filename)
    sim_info["states_shape"] = states_shape
    sim_info["mask_shape"] = mask_shape
    sim_info["states_path"] = states_path
    sim_info["packed_mask_path"] = packed_mask_path


def ensure_sim_cache(sim_info):
    sim_name = os.path.basename(sim_info["dir"])
    if _validate_sim_cache(sim_info):
        print(f"Using existing cache for {sim_name}", flush=True)
        return "existing"

    _build_sim_cache(sim_info)
    return "rebuilt"


def _ensure_sim_cache_worker(sim_info):
    return ensure_sim_cache(sim_info)


def ensure_all_sim_caches(sim_infos, cache_workers, states_filename, mask_filename):
    for sim_info in sim_infos:
        prepare_sim_cache_info(sim_info, states_filename, mask_filename)

    worker_count = min(cache_workers, len(sim_infos))
    if worker_count <= 1:
        for sim_info in sim_infos:
            ensure_sim_cache(sim_info)
        return

    with get_context("spawn").Pool(processes=worker_count) as pool:
        for _ in pool.imap_unordered(_ensure_sim_cache_worker, sim_infos, chunksize=1):
            pass

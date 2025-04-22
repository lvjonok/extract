import os
import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image

imwidth = 128
imheight = 128


def append_data(h5_name, to_add, key_name):
    with h5py.File(h5_name, "a") as f:
        f[key_name].resize((len(to_add) + f[key_name].shape[0]), axis=0)
        f[key_name][-len(to_add) :] = np.stack(to_add)


dt = h5py.special_dtype(vlen=str)
base_path = "./datasets/"

# TODO: libero_90 was removed for now
splits = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]

if not os.path.exists(os.path.join(base_path, "processed_libero_dataset")):
    os.mkdir(os.path.join(base_path, "processed_libero_dataset"))

for type_split in splits:
    hdf5_name = os.path.join(
        base_path, "processed_libero_dataset", type_split + ".hdf5"
    )
    rendered_frames = np.empty((0, imwidth, imheight, 3), dtype=np.uint8)
    rel_action = np.empty((0, 7), dtype=np.float32)
    terminal = np.empty((0, 1), dtype=np.uint8)
    with h5py.File(hdf5_name, "w") as f:
        f.create_dataset(
            "actions",
            data=rel_action,
            maxshape=(None, 7),
            compression="gzip",
        )
        f.create_dataset(
            "terminals",
            data=terminal,
            maxshape=(None, 1),
            compression="gzip",
        )
        f.create_dataset(
            "rendered_frames",
            data=np.array(rendered_frames),
            maxshape=(None, imwidth, imheight, 3),
            compression="gzip",
        )
    file_paths = os.path.join(base_path, type_split)
    files = os.listdir(file_paths)
    files = [file for file in files if "hdf5" in file]
    for file in tqdm(files):
        curr_path = os.path.join(file_paths, file)
        with h5py.File(curr_path) as scene_demos:
            for demo in scene_demos["data"].keys():
                actions = scene_demos["data"][demo]["actions"][()]
                terminals = scene_demos["data"][demo]["dones"][()][:, None]
                rendered_frames = scene_demos["data"][demo]["obs"]["agentview_rgb"][()]
                if (
                    rendered_frames.shape[1] != imwidth
                    or rendered_frames.shape[2] != imheight
                ):
                    # resize rendered frames to imwidth, imheight
                    rendered_frames = np.array(
                        Image.fromarray(rendered_frames).resize((imwidth, imheight))
                    ).astype(np.uint8)

                # flip the agentview_rgb images over the X axis
                rendered_frames = rendered_frames[:, ::-1, :, :]

                append_data(hdf5_name, actions, "actions")
                append_data(hdf5_name, terminals, "terminals")
                append_data(hdf5_name, rendered_frames, "rendered_frames")

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import h5py
import json
from sentence_transformers import SentenceTransformer

imwidth = 84
imheight = 84

base_path = "./datasets/"
model = SentenceTransformer("all-MiniLM-L6-v2")

# helper to create or append a dataset
def append_data(h5_path, key, data):
    with h5py.File(h5_path, "a") as f:
        if key not in f:
            maxshape = (None,) + data.shape[1:]
            f.create_dataset(
                key, data=data, maxshape=maxshape, dtype=data.dtype, compression="gzip"
            )
        else:
            ds = f[key]
            old = ds.shape[0]
            new = old + data.shape[0]
            ds.resize((new,) + ds.shape[1:])
            ds[old:new] = data

splits = ["libero_90", "libero_10", "libero_goal", "libero_object", "libero_spatial"]

if not os.path.exists(os.path.join(base_path, "processed_libero_dataset_lowres")):
    os.mkdir(os.path.join(base_path, "processed_libero_dataset_lowres"))

for type_split in splits:
    # build one .h5 per split
    output_file = os.path.join(
        base_path, "processed_libero_dataset_lowres", f"{type_split}.h5"
    )
    if os.path.exists(output_file):
        os.remove(output_file)

    file_paths = os.path.join(base_path, type_split)
    files = [f for f in os.listdir(file_paths) if f.endswith(".hdf5")]
    for file in tqdm(files):
        curr_path = os.path.join(file_paths, file)
        with h5py.File(curr_path, "r") as scene_demos:
            lang_instr = json.loads(scene_demos["data"].attrs["problem_info"])[
                "language_instruction"
            ]
            embedded_lang = model.encode([lang_instr])
            for demo in scene_demos["data"].keys():
                actions = scene_demos["data"][demo]["actions"][()].astype(np.float32)
                terminals = scene_demos["data"][demo]["dones"][()][:, None].astype(np.uint8)
                rendered_frames = scene_demos["data"][demo]["obs"]["agentview_rgb"][()]

                # resize if needed
                if (
                    rendered_frames.shape[1] != imwidth
                    or rendered_frames.shape[2] != imheight
                ):
                    rendered_frames = np.stack(
                        [
                            np.array(
                                Image.fromarray(frame).resize((imwidth, imheight))
                            )
                            for frame in rendered_frames
                        ]
                    ).astype(np.uint8)

                # flip over X axis
                rendered_frames = rendered_frames[:, ::-1, :, :].astype(np.uint8)

                # append each array
                append_data(output_file, "actions", actions)
                append_data(output_file, "terminals", terminals)
                append_data(output_file, "rendered_frames", rendered_frames)
                # tile lang embeds to match frames
                lang_batch = np.tile(embedded_lang.astype(np.float16), (actions.shape[0], 1))
                append_data(output_file, "lang_embeds", lang_batch)

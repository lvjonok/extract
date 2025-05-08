import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import pyxis as px
import h5py
import json

imwidth = 84
imheight = 84

base_path = "./datasets/"
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# TODO: libero_90 was removed for now
splits = ["libero_90", "libero_10", "libero_goal", "libero_object", "libero_spatial"]

if not os.path.exists(os.path.join(base_path, "processed_libero_dataset_lowres")):
    os.mkdir(os.path.join(base_path, "processed_libero_dataset_lowres"))

for type_split in splits:
    file_name = os.path.join(base_path, "processed_libero_dataset_lowres", type_split)
    with px.Writer(
        dirpath=file_name,
        map_size_limit=40000,
    ) as db:
        file_paths = os.path.join(base_path, type_split)
        files = os.listdir(file_paths)
        files = [file for file in files if "hdf5" in file]
        for file in tqdm(files):
            curr_path = os.path.join(file_paths, file)
            with h5py.File(curr_path) as scene_demos:
                lang_instr = json.loads(scene_demos["data"].attrs["problem_info"])[
                    "language_instruction"
                ]
                embedded_lang = model.encode([lang_instr])
                for demo in scene_demos["data"].keys():
                    actions = scene_demos["data"][demo]["actions"][()]
                    terminals = scene_demos["data"][demo]["dones"][()][:, None]
                    rendered_frames = scene_demos["data"][demo]["obs"]["agentview_rgb"][
                        ()
                    ]

                    if (
                        rendered_frames.shape[1] != imwidth
                        or rendered_frames.shape[2] != imheight
                    ):
                        # resize rendered frames to imwidth, imheight
                        rendered_frames = np.stack(
                            [
                                np.array(
                                    Image.fromarray(frame).resize((imwidth, imheight))
                                )
                                for frame in rendered_frames
                            ]
                        ).astype(np.uint8)

                    # flip the agentview_rgb images over the X axis
                    rendered_frames = rendered_frames[:, ::-1, :, :]

                    to_be_put_in_db = []
                    to_be_put_in_db.extend(["actions", actions.astype(np.float32)])
                    to_be_put_in_db.extend(["terminals", terminals.astype(np.uint8)])
                    to_be_put_in_db.extend(
                        ["rendered_frames", rendered_frames.astype(np.uint8)]
                    )
                    to_be_put_in_db.extend(
                        [
                            "lang_embeds",
                            np.tile(
                                embedded_lang.astype(np.float16), (actions.shape[0], 1)
                            ),
                        ]
                    )
                    db.put_samples(*to_be_put_in_db)

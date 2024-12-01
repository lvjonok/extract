# this file views frames from a specific cluster file
import argparse
import tqdm
import h5py
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

env_name_to_path = {
    "kitchen": "spirl_prim/datasets/",
    "calvin": "calvin/dataset/task_ABCD_D/calvin_dataset",
    "libero": "LIBERO/libero/datasets/processed_libero_dataset/",
}

final_folder_to_dataset_file = {
    "training": "training.hdf5",
    "validation": "validation.hdf5",
    "10": "libero_10.hdf5",
    "90": "libero_90.hdf5",
    "spatial": "libero_spatial.hdf5",
    "goal": "libero_goal.hdf5",
    "object": "libero_object.hdf5",
    "mixed": "kitchen_mixed_data.h5",
}


def animate_clusters(args):
    cluster_file = args.cluster_file
    num_frames = args.num_frames
    num_samples_per_cluster = args.num_samples_per_cluster

    with h5py.File(cluster_file, "r") as f:
        clusters = f["clusters"][()]

    # iterate through and map all clusters to their respective indices
    cluster_index_map = defaultdict(list)
    for i in range(len(clusters)):
        cluster_index_map[clusters[i]].append(i)

    # grab the correct frames sampling num_frames, num_samples_per_cluster times
    cluster_frame_map = defaultdict(list)

    # get the dataset_file
    for env_name in env_name_to_path.keys():
        if env_name in cluster_file:
            dataset_file_dir = env_name_to_path[env_name]
            break

    dataset_file = f"{dataset_file_dir}/{args.dataset_file_name}"
    with h5py.File(dataset_file, "r") as f:
        frames = f["rendered_frames"]
        for cluster, indices in cluster_index_map.items():
            for i in range(num_samples_per_cluster):
                # pick a random index from the indices
                if len(indices) < num_frames:
                    random_index = 0
                else:
                    random_index = np.random.randint(0, len(indices) - num_frames + 1)
                # grab the frames
                cluster_frame_map[cluster].append(
                    frames[indices[random_index : random_index + num_frames]]
                )
                # add 5 black frames to signify end
                size = cluster_frame_map[cluster][-1].shape[1]
                cluster_frame_map[cluster][-1] = np.concatenate(
                    [
                        cluster_frame_map[cluster][-1],
                        np.zeros((5, size, size, 3), dtype=np.uint8),
                    ]
                )
            cluster_frame_map[cluster] = np.concatenate(cluster_frame_map[cluster])
    save_path = cluster_file.replace("/clusters.h5", "")
    os.makedirs(f"{save_path}/cluster_frames", exist_ok=True)
    print("Plotting the per length histograms and animating")
    for skill in tqdm.tqdm(cluster_frame_map.keys()):
        # save the frames as a video
        print("ANIMATING")
        # get the frames
        skill_frames = cluster_frame_map[skill]
        # sample 3 500 frame intervals at random
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        fps = 40
        video_filename = f"{save_path}/cluster_frames/{skill}.avi"
        out = cv2.VideoWriter(
            video_filename,
            fourcc,
            fps,
            (skill_frames.shape[1], skill_frames.shape[2]),
        )
        for frame in skill_frames:
            # convert to bgr
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        out.release()


# let's do this for all
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cluster_file", type=str, required=True)
    # parser.add_argument("--dataset_file_name", type=str, required=True)
    # parser.add_argument("--num_frames", type=int, default=100)
    # parser.add_argument("--num_samples_per_cluster", type=int, default=5)
    # args = parser.parse_args()
    envs = ["kitchen", "libero"]
    root_folder = "generated_dataset_clusters"
    for env in envs:
        path = os.path.join(root_folder, env)
        for method in os.listdir(path):
            before_train_val = os.path.join(path, method)
            for final_folder in os.listdir(before_train_val):
                files_in_this_path = os.listdir(
                    os.path.join(before_train_val, final_folder)
                )
                if (
                    "clusters.h5" in files_in_this_path
                    and "cluster_frames" not in files_in_this_path
                ):
                    cluster_file = os.path.join(
                        before_train_val, final_folder, "clusters.h5"
                    )
                    dataset_file_name = final_folder_to_dataset_file[final_folder]
                    args = argparse.Namespace(
                        cluster_file=cluster_file,
                        dataset_file_name=dataset_file_name,
                        num_frames=100,
                        num_samples_per_cluster=5,
                    )

                    animate_clusters(args)

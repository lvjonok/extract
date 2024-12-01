import os

default_n_threads = 8
os.environ["OPENBLAS_NUM_THREADS"] = f"{default_n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{default_n_threads}"
os.environ["OMP_NUM_THREADS"] = f"{default_n_threads}"

import pandas as pd
import h5py
import sys

# import ffmpeg
import imageio.v3 as iio
import cv2
import pickle
from threading import Thread
from collections import defaultdict
from sklearn.base import ClusterMixin
from sklearn import cluster
from sklearn import mixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# from trendypy.trendy import Trendy
from matplotlib import animation
from vlm_reward_funcs import *
import scipy
import tqdm
from vlm_tests import readGif, construct_model
import seaborn as sns
import matplotlib.pyplot as plt

# from torch import compile

""" 
This script will embed each timestep of the entire dataset input sequence and then cluster the embeddings. These embeddings will be saved to a separate file to indicate a cluster index for each of the datasets.

Do this over multiple clustering hyperparameters and algorithms and then save them to all different folders, along with results to one csv file

1. Load the model and all of the frames and generate the embeddings
2. Cluster the embeddings
3. Process the clustering using a median filter so that the clusters are more stable (optional)
4. Label the embeddings with skills based on clustering and embeddings. Each skill is one continuous cluster segment until reaching a hard boundary like the end of a trajectory.
5. (Optionally) visualize each of the clusters in sequence and save the videos to a folder if ANIMATE=True

"""

DEBUG = False
ANIMATE = False
PLOT_SKILLS = True
frame_res = 128
PCA_RATIO = 0.99

class Filter:
    def __init__(self, filter_params, filter):
        self.filter_params = filter_params
        self.filter = filter

    def __call__(self, labels):
        return self.filter(labels, **self.filter_params)


class MedianFilter(Filter):
    def __init__(self, window_size, skip_length):
        super().__init__(
            {"window_size": window_size, "skip_length": skip_length}, median_filter
        )

    def __str__(self):
        return f"MedFilt({self.filter_params['window_size'], self.filter_params['skip_length']})"


class ScipyMedianFilter(Filter):
    def __init__(self, kernel_size):
        super().__init__({"kernel_size": kernel_size}, scipy.signal.medfilt)

    def __str__(self):
        return f"MedFilt({self.filter_params['kernel_size']})"


class NoFilter(Filter):
    def __init__(self):
        super().__init__({}, lambda x: x)

    def __str__(self):
        return f"NoFilt()"


def median_filter(
    labels,
    window_size,
    skip_length,
):
    # median filter the labels
    for i in range(window_size // 2, len(labels) - (window_size // 2), skip_length):
        window = labels[i - window_size // 2 : i + window_size // 2]
        labels[i] = np.median(window)
    return labels

def cluster_frames(embeddings_list, clustering_model: ClusterMixin):
    # bad code
    train = embeddings_list[0]
    if len(embeddings_list) == 2:
        val = embeddings_list[1]
        flattened_val_embeddings = torch.cat(val, dim=0)
    elif len(embeddings_list) > 2:
        vals = embeddings_list[1:]
        flattened_val_embeddings = []
        for val in vals:
            flattened_val_embeddings.append(torch.cat(val, dim=0))
        flattened_val_embeddings = torch.cat(flattened_val_embeddings, dim=0)
    flattened_train_embeddings = torch.cat(train, dim=0)
    print("Clustering...")
    # pca = PCA(n_components=PCA_RATIO, svd_solver="full")
    # projected_train_embeddings = torch.from_numpy(pca.fit_transform(flattened_train_embeddings))
    # print(f"PCA Projected to {100 * PCA_RATIO}% explained variance ratio, n = {pca.n_components_}")
    train_labels = clustering_model.fit_predict(flattened_train_embeddings)
    # train_labels = clustering_model.fit_predict(projected_train_embeddings)
    if len(embeddings_list) >= 2:
        val_labels = clustering_model.predict(flattened_val_embeddings)
        # projected_val_embeddings = torch.from_numpy(pca.transform(flattened_val_embeddings))
        # val_labels = clustering_model.predict(projected_val_embeddings)

    if len(embeddings_list) >= 2:
        labels = np.concatenate((train_labels, val_labels))
    else:
        labels = train_labels
    # cast all -1 to a new one
    labels[labels == -1] = labels.max() + 1

    print("fitting logprobs")
    # fit a gaussian model to the clusterings by MLE estimation for each cluster
    per_cluster_params = {}
    for label in range(clustering_model.n_clusters):
        # get the embeddings for this cluster
        cluster_embeddings = flattened_train_embeddings[train_labels == label]
        # cluster_embeddings = projected_train_embeddings[train_labels == label]
        # fit a gaussian model
        mean = torch.mean(cluster_embeddings, dim=0)
        std = torch.sqrt(torch.var(cluster_embeddings.float(), dim=0))
        per_cluster_params[label] = (mean, std)

    # now compute the logprobs for each of the embeddings
    per_cluster_logprobs = []
    if len(embeddings_list) >= 2:
        all_embeddings = torch.cat(
            (flattened_train_embeddings, flattened_val_embeddings)
        )
        # all_embeddings = torch.cat((projected_train_embeddings, projected_val_embeddings))
    else:
        all_embeddings = flattened_train_embeddings
        # all_embeddings = projected_train_embeddings
    ####
    # all_embeddings = flattened_train_embeddings
    # import pdb; pdb.set_trace()
    ####
    for label in range(clustering_model.n_clusters):
        mean, std = per_cluster_params[label]
        logprob = (
            torch.distributions.Normal(
                mean.float(), torch.clamp(std.float(), min=1e-10)
            )
            .log_prob(all_embeddings.float())
            .cpu()
        )
        per_cluster_logprobs.append(logprob.mean(-1))
    per_cluster_logprobs = torch.stack(per_cluster_logprobs)

    # log_softmax
    # per_cluster_logprobs_softmax = torch.nn.functional.log_softmax(
    #    per_cluster_logprobs.float(), dim=0
    # )
    if torch.any(torch.isnan(per_cluster_logprobs)):
        import pdb

        pdb.set_trace()
    return labels, per_cluster_logprobs.numpy()


def plot_cluster_video(
    skill_frames: list,
    skill_embeddings: list,
    skill_labels: list,
    save_path: str,
    logprobs: list,
):
    # frames is a list of tensors for frames
    # skill labels is a list of labels for each of the list of frames
    # embeddings is a list of embeddings for each of the list of frames
    # plot the clusters
    # sns.set_style("whitegrid", {"axes.grid": True})

    # flatten frames list and embeddings list and make skill_labels the same length
    # frames may have different shapes each
    flattened_frames = []
    for frames in skill_frames:
        flattened_frames += list(frames)
    flattened_embeddings = torch.cat(skill_embeddings)
    flattened_skill_labels = []
    skill_to_length_map = defaultdict(list)
    all_lengths = []
    for skill_label_list in skill_labels:
        flattened_skill_labels += skill_label_list
        # get the lengths of each skill computed by iterating through and checking when it changes
        current_length = 1
        for i in range(len(skill_label_list) - 1):
            if skill_label_list[i] != skill_label_list[i + 1]:
                skill_to_length_map[skill_label_list[i - 1]].append(current_length)
                all_lengths.append(current_length)
                current_length = 1
            else:
                current_length += 1
        # last one
        skill_to_length_map[skill_label_list[-1]].append(current_length)
        all_lengths.append(current_length)
    if not tuple(all_lengths) == tuple([len(em) for em in skill_embeddings]):
        import pdb

        pdb.set_trace()

    sorted_flattened_skill_labels, sorted_indicies = torch.sort(
        torch.tensor(flattened_skill_labels)
    )
    sorted_flattened_skill_labels_list = sorted_flattened_skill_labels.tolist()

    # skill labels is a list of labels for each of the list of frames
    # we'll sort the frames and embeddings by skill labels in increasing order
    flattened_frames = [flattened_frames[i] for i in sorted_indicies]
    flattened_embeddings = flattened_embeddings[sorted_indicies]

    # pca to 3d
    # pca = PCA(n_components=3)
    # projected_flattened_embeddings = pca.fit_transform(flattened_embeddings)

    # make a color map for the original skill labels
    # use distinct cmap
    # unique_skill_labels = list(set(sorted_flattened_skill_labels_list))
    # cmap = plt.cm.get_cmap("tab20", len(unique_skill_labels))
    # color_map = dict(zip(unique_skill_labels, cmap.colors))

    # calculate silhouette score to measure cluster quality
    try:
        non_projected_score = silhouette_score(
            flattened_embeddings, sorted_flattened_skill_labels_list
        )
        projected_score = 0
    except ValueError:
        # if there's only one cluster, then the silhouette score will throw an error
        non_projected_score = 0
        projected_score = 0

    sns.set(font_scale=1.5)
    # check if it exists already
    if not (os.path.exists(f"{save_path}/skill_lengths.png")):
        # tight boundaries
        os.makedirs(save_path, exist_ok=True)
        if PLOT_SKILLS:
            # plot 2 plots of the histograms of skill lengths generally and the avg length and std of each skill
            fig, axs = plt.subplots(3, 1, figsize=(12, 11))
            plt.subplots_adjust(hspace=0.3)
            # plot the histogram of skill lengths

            all_lengths = np.array(
                [
                    length
                    for skill in skill_to_length_map
                    for length in skill_to_length_map[skill]
                ]
            )
            axs[0].hist(
                all_lengths, bins=np.arange(np.max(all_lengths) + 1), density=True
            )
            axs[0].set_title("Histogram of Skill Lengths")
            ## plot the box plot of each skill
            # axs[1].bar(
            #    skill_to_length_map.keys(),
            #    [np.mean(skill_to_length_map[skill]) for skill in skill_to_length_map],
            #    yerr=[np.std(skill_to_length_map[skill]) for skill in skill_to_length_map],
            # )

            # get the labels in order
            ordered_labels = sorted(list(skill_to_length_map.keys()))
            ordered_values = [skill_to_length_map[skill] for skill in ordered_labels]
            axs[1].boxplot(
                ordered_values,
                labels=ordered_labels,
                showmeans=True,
                meanline=True,
            )
            axs[1].set_title(
                f"Box Plot of Skill Lengths -- Avg Overall Length: {np.mean([length for skill in skill_to_length_map for length in skill_to_length_map[skill]]):0.1f}"
            )
            axs[1].set_ylabel("Length of Skill")

            # now plot on the last axes an overlapping histplot of the skill lengths per cluster
            # first make a dataframe
            skill_lengths_df = pd.DataFrame()
            for skill, lengths in skill_to_length_map.items():
                skill_lengths_df = skill_lengths_df.append(
                    pd.DataFrame(
                        {
                            "skill": [skill for _ in range(len(lengths))],
                            "length": lengths,
                        }
                    )
                )
            # plot a subsample of the data
            print("Plotting the entire df histogram")
            skill_lengths_df = skill_lengths_df.sample(min(1000, len(skill_lengths_df)))
            sns.histplot(
                data=skill_lengths_df,
                x="length",
                hue="skill",
                element="step",
                ax=axs[2],
            )
            axs[2].set_title("Skill Lengths per Cluster")
            # make the legend multiple columns
            # handles, labels = axs[2].get_legend_handles_labels()
            axs[2].legend()
            # axs[2].legend(
            #    handles=handles,
            #    labels=labels,
            #    loc="best",
            #    ncol=2,
            #    fontsize="small",
            #    fancybox=True,
            # )
            plt.savefig(f"{save_path}/skill_lengths.pdf", bbox_inches="tight")

            # now save the frames, in sequence, from each cluster into a folder containing videos of all frames in sequence for each cluster
            if ANIMATE:
                os.makedirs(f"{save_path}/cluster_frames", exist_ok=True)
            print("Plotting the per length histograms and animating")
            for skill in tqdm.tqdm(ordered_labels):
                # get the indices of the frames that belong to this skill
                skill_indicies = sorted_flattened_skill_labels[
                    sorted_flattened_skill_labels == skill
                ]
                # skill_indicies = [
                #    i for i, x in enumerate(sorted_flattened_skill_labels) if x == skill
                # ]

                # save the frames as a video
                if ANIMATE:
                    print("ANIMATING")
                    # get the frames
                    skill_frames = [flattened_frames[i] for i in skill_indicies]
                    # sample 3 500 frame intervals at random
                    if len(skill_frames) > 1000:
                        new_skill_frames = []
                        for i in range(3):
                            start = np.random.randint(0, len(skill_frames) - 500)
                            new_skill_frames += skill_frames[start : start + 500]
                        skill_frames = new_skill_frames
                    skill_frames = torch.stack(skill_frames).numpy()
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

                # now also save a per-skill histogram plot
                # first clear the plot
                print(f"Saving per-skill histogram for cluster {skill}")
                plt.clf()
                # plot a subsample of skill_to_length_map
                skill_length_sample = np.random.choice(
                    skill_to_length_map[skill], size=500, replace=True
                )
                plt.hist(
                    skill_length_sample,
                    bins=np.arange(np.max(all_lengths) + 1),
                    density=True,
                )
                plt.title(f"Histogram of Lengths for Cluster {skill}")
                plt.savefig(
                    f"{save_path}/cluster_lengths_{skill}.pdf", bbox_inches="tight"
                )
        print("Done with plotting")
    # save the cluster
    save_generated_clusters(
        flattened_skill_labels, logprobs, f"{save_path}/clusters.h5"
    )
    # save the clustering model params
    with open(f"{save_path}/clustering_model.pkl", "wb") as f:
        pickle.dump(clustering_model, f)
    return (
        non_projected_score,
        projected_score,
    )


def load_and_label_vids_in_saved_dataset(
    vlm: VLMRewardFunction, hdf5_paths: list[str], animate=False
):
    # load the videos in the hdf5 and label them with the vlm
    # returns a list of frames, a list of skill labels, and a list of embeddings
    # use tqdm
    all_hdf5_frames = []
    all_hdf5_embeddings = []
    hdf5_names = []
    print("Loading videos and computing embeddings...")
    for hdf5_path in hdf5_paths:
        frames_list = []
        embeddings_list = []
        try:
            hdf5_names.append(hdf5_path.split("/")[-1].split(".")[0].split("_")[1])
        except:
            hdf5_names.append(hdf5_path.split("/")[-1].split(".")[0])
        with h5py.File(hdf5_path, "r") as h5_dataset:
            all_frames = h5_dataset["rendered_frames"]
            # split dataset into sequences
            seq_end_idxs = (
                np.where(h5_dataset["terminals"][1:])[0] + 1
            )  # don't count  0'th if it's a terminal (CALVIN data format issue)
            # also make sure the last one is included
            if len(all_frames) - 1 not in seq_end_idxs:
                seq_end_idxs = np.concatenate(
                    (seq_end_idxs, np.array([len(all_frames) - 1]))
                )
            start = 0
            print("Loading Data")
            # TODO: check if there's an off by 1 error here with respect to the actions
            if DEBUG:
                seq_end_idxs = seq_end_idxs[:2]
            for end_idx in tqdm.tqdm(seq_end_idxs):
                frames = torch.from_numpy(all_frames[start : end_idx + 1])
                embeddings = vlm.compute_video_embed(frames)
                if embeddings.shape[0] == frames.shape[0] - 1:
                    embeddings = torch.cat((embeddings[0:1], embeddings), dim=0)
                # downsize frame if needed
                if animate:
                    if frames.shape[2] != frame_res:
                        frames = torch.stack(
                            [
                                torch.from_numpy(
                                    cv2.resize(
                                        frame.numpy(),
                                        (
                                            frame_res,
                                            frame_res,
                                        ),
                                        interpolation=cv2.INTER_AREA,
                                    )
                                )
                                for frame in frames
                            ]
                        )
                else:
                    frames = torch.empty(embeddings.shape[0])
                frames_list.append(frames)
                embeddings_list.append(embeddings)
                start = end_idx + 1
        all_hdf5_frames.append(frames_list)
        all_hdf5_embeddings.append(embeddings_list)
    return all_hdf5_frames, all_hdf5_embeddings, hdf5_names


def filter_embeddings(
    labels: np.array,
    embeddings_list: list,
    frames_list: list,
    filter: Filter,
):
    # now label the embeddings with skills based on clustering and embeddings
    # each skill is one continuous cluster segment until reaching a hard boundary like the end of a trajectory or a different skill belonging to another cluster label
    all_skill_frames = []
    all_skill_embeddings = []
    all_filtered_labels = []

    curr_label_index = 0
    index_correspondence = (
        []
    )  # to keep track of which skill assignments are from which hdf5 datasets from frames_list/embeddings_list
    print("Filtering Embeddings")
    for i, embeddings in enumerate(tqdm.tqdm(embeddings_list)):
        skill_frames = []
        skill_embeddings = []
        filtered_labels = filter(
            labels[curr_label_index : curr_label_index + len(embeddings)]
        )
        all_filtered_labels.append(list(filtered_labels))
        curr_label_index += len(embeddings)
        for j, embedding in enumerate(embeddings):
            if j == 0:
                skill_frames.append(frames_list[i][j])
                skill_embeddings.append(embedding)
            elif filtered_labels[j] == filtered_labels[j - 1]:
                skill_frames.append(frames_list[i][j])
                skill_embeddings.append(embedding)
            else:
                all_skill_frames.append(torch.stack(skill_frames))
                all_skill_embeddings.append(torch.stack(skill_embeddings))
                index_correspondence.append(i)
                skill_frames = [frames_list[i][j]]
                skill_embeddings = [embedding]
        all_skill_frames.append(torch.stack(skill_frames))
        all_skill_embeddings.append(torch.stack(skill_embeddings))
        index_correspondence.append(i)
    assert (
        len(all_skill_frames) == len(all_skill_embeddings) == len(index_correspondence)
    )
    return (
        all_skill_frames,
        all_skill_embeddings,
        all_filtered_labels,
        index_correspondence,
    )


def video_load_worker(env_name, total_paths, model, list: list, animate=False):
    (frames_list, embeddings_list, hdf5_names) = load_and_label_vids_in_saved_dataset(
        model, total_paths, animate=animate
    )
    list.append((env_name, frames_list, embeddings_list, hdf5_names))


def save_generated_clusters(cluster_list, logprobs, save_path):
    print("Saving Clusters")
    with h5py.File(save_path, "w") as f:
        f.create_dataset("clusters", data=cluster_list, dtype=np.uint8)
        f.create_dataset("logprobs", data=logprobs)
    print("Clusters saved")


if __name__ == "__main__":
    # get first arg to check if we're doing kitchen or libero
    dataset = sys.argv[1]
    env_name_to_dataset_names = {
        "kitchen": ["kitchen"],
        "libero": [
            "libero_90",
            "libero_10",
            "libero_goal",
            "libero_object",
            "libero_spatial",
        ],
    }
    assert dataset in env_name_to_dataset_names, f"Dataset {dataset} not found"
    which_envs = env_name_to_dataset_names[dataset]
    env_names = [env.split("_")[0] for env in which_envs]
    assert len(set(env_names)) == 1
    env_name = env_names[0]

    env_name_to_path = {
        "kitchen": "./datasets/kitchen_mixed_data.h5",
        "libero": "./datasets/processed_libero_dataset/",
    }
    save_path = "./generated_dataset_clusters"

    param_groups = {
        # "xi": [cluster.OPTICS],
        # "damping": [cluster.AffinityPropagation],
        "n_clusters": [
            cluster.MiniBatchKMeans,
            # cluster.KMeans,
            # cluster.SpectralClustering,
            # cluster.AgglomerativeClustering,
            # Trendy,
        ],
        # "n_components": [mixture.GaussianMixture],
        # "": [cluster.MeanShift],
    }

    # param_combinations = {"n_clusters": [3, 5, 8, 12, 15], "xi": [0.01, 0.05, 0.2]}
    param_combinations = {"n_clusters": [8], "xi": [0.01, 0.05, 0.2]}
    # param_combinations = {"n_clusters": [8], "xi": [0.01, 0.05, 0.2]}

    alg_to_cls_name = {
        cluster.DBSCAN: "DBSCAN",
        cluster.OPTICS: "Optics",
        cluster.AffinityPropagation: "AffinityProp",
        cluster.MiniBatchKMeans: "KMeans",
        cluster.KMeans: "KMeans",
        cluster.SpectralClustering: "Spectral",
        cluster.AgglomerativeClustering: "Agglomerative",
        mixture.GaussianMixture: "GMM",
        cluster.MeanShift: "MeanShift",
    }

    reward_types = ["first_diff"]
    # reward_types = ["state"]
    # reward_types=["first_diff"]
    # reward_types = [
    #    "first_diff",
    # ]  # real_diff"]

    # save the score to a pandas dataframe
    results_csv = f"{save_path}/results_{env_name}.csv"
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
    else:
        # do this for each of the clustering algorithms
        # env_names = pkl_paths
        env_result_choices = ["", " proj"]
        env_result_columns = [
            f"{name}{choice}" for choice in env_result_choices for name in which_envs
        ]
        avgs = ["avg"]
        df = pd.DataFrame(
            columns=[
                "vlm",
                *env_result_columns,
                *avgs,
            ]
        )
    cluster_filters = [
        # NoFilter(),
        # ScipyMedianFilter(5),
        ScipyMedianFilter(7),
        # ScipyMedianFilter(9),
        # ScipyMedianFilter(11),
    ]
    vlm_choices = [
        # "state_diff_ablation",
        "r3m-99",
        # "LIV",
        # "kitchen_proprioceptive",
        # "openai/clip-vit-base-patch16",
        # "s3d",
        # "microsoft/xclip-base-patch16-zero-shot",
        # "microsoft/xclip-large-patch14",
    ]
    for model_name in vlm_choices:
        for reward_type in reward_types:
            vlm_config = AttrDict()
            vlm_config.vlm = model_name
            vlm_config.device = "cuda:0"
            vlm_config.reward_type = reward_type
            model = construct_model(vlm_config)
            # model = compile(model, mode="reduce-overhead")
            cleaned_model_name = f"{model.config.vlm.split('/')[-1]}"
            if hasattr(model, "reward_type"):
                cleaned_model_name += f"_{model.reward_type}"
            loaded_and_labeled_vids = []
            # get all hdf5's in the env_name_to_path[env_name]
            if env_name == "kitchen":
                h5_paths = [env_name_to_path[env_name]]
            else:
                h5_paths = os.listdir(env_name_to_path[env_name])
                h5_paths = [
                    os.path.join(env_name_to_path[env_name], h5_path)
                    for h5_path in h5_paths
                    if h5_path.endswith(".hdf5")
                ]
            video_load_worker(
                env_name,
                h5_paths,
                model,
                loaded_and_labeled_vids,
                animate=ANIMATE,
            )
            # threads = []
            # do this with multiple threads up to MAX_WORKERS
            # for pkl_path in pkl_paths:
            #    total_path = os.path.join(data_path, pkl_path)
            #    thread = Thread(
            #        target=video_load_worker,
            #        args=(pkl_path, total_path, model, loaded_and_labeled_vids),
            #    )
            #    thread.start()
            #    threads.append(thread)
            # for thread in threads:
            #    thread.join()

            for param_group in param_groups:
                for param_value in param_combinations[param_group]:
                    for clustering_cls in param_groups[param_group]:
                        if param_group == "":
                            clustering_model = clustering_cls()
                        else:
                            kwargs = {param_group: param_value}
                        # if alg_to_cls_name[clustering_cls] == "KMeans":
                        #    kwargs["batch_size"] = 256 * 8
                        clustering_model = clustering_cls(**kwargs)
                        for (
                            env_name,
                            all_frames_list,
                            all_embeddings_list,
                            hdf5_names,
                        ) in loaded_and_labeled_vids:
                            assert len(hdf5_names) == len(which_envs)
                            # cluster the embeddings
                            # make a joined list first
                            joined_frames_list = []
                            joined_embeddings_list = []
                            # bad code
                            if env_name == "kitchen":
                                assert (
                                    len(all_frames_list)
                                    == len(all_embeddings_list)
                                    == 1
                                )
                            for frames_list, embeddings_list in zip(
                                all_frames_list, all_embeddings_list
                            ):
                                joined_frames_list.extend(frames_list)
                                joined_embeddings_list.extend(embeddings_list)
                            labels, logprobs = cluster_frames(
                                all_embeddings_list, clustering_model
                            )
                            for cluster_filter in cluster_filters:
                                print(f"Filtering with {cluster_filter}")
                                clustering_model_name = f"{alg_to_cls_name[clustering_cls]}_{param_group}:{param_value}_{cluster_filter}"
                                save_name = (
                                    f"{cleaned_model_name}_{clustering_model_name}"
                                )
                                scores = []
                                rand_scores = []
                                total_save_path = os.path.join(
                                    save_path,
                                    env_name,
                                    f"{cleaned_model_name}_{clustering_model_name}",
                                )
                                (
                                    skill_frames,
                                    skill_embeddings,
                                    skill_labels,
                                    index_correspondence,
                                ) = filter_embeddings(
                                    np.copy(labels),
                                    joined_embeddings_list,
                                    joined_frames_list,
                                    cluster_filter,
                                )
                                # now for each of them plot the clusters
                                for i in range(len(all_frames_list)):
                                    skill_label_start = sum(
                                        [len(all_frames_list[j]) for j in range(i)]
                                    )
                                    skill_label_end = skill_label_start + len(
                                        all_frames_list[i]
                                    )
                                    indicies_to_keep = [
                                        j
                                        for j, x in enumerate(index_correspondence)
                                        if x >= skill_label_start
                                        and x < skill_label_end
                                    ]
                                    curr_skill_frames = [
                                        frames
                                        for j, frames in enumerate(skill_frames)
                                        if j in indicies_to_keep
                                    ]
                                    curr_skill_embeddings = [
                                        embeddings
                                        for j, embeddings in enumerate(skill_embeddings)
                                        if j in indicies_to_keep
                                    ]
                                    curr_skill_labels = skill_labels[
                                        skill_label_start:skill_label_end
                                    ]
                                    curr_total_save_path = os.path.join(
                                        total_save_path, hdf5_names[i]
                                    )
                                    curr_logprobs = logprobs[
                                        skill_label_start:skill_label_end
                                    ]
                                    (
                                        env_non_projected_score,
                                        env_projected_score,
                                    ) = plot_cluster_video(
                                        curr_skill_frames,
                                        curr_skill_embeddings,
                                        curr_skill_labels,
                                        curr_total_save_path,
                                        curr_logprobs,
                                    )
                                    scores.extend(
                                        [
                                            env_non_projected_score,
                                            env_projected_score,
                                        ]
                                    )
                            df.loc[len(df)] = [
                                save_name,
                                *scores,
                                np.mean(scores),
                            ]
                            df.to_csv(results_csv, index=False)

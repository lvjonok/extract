import PIL
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import skvideo.io
from typing import List
from vlm_reward_funcs import (
    VLMRewardFunction,
    StateDiffRewardFunction,
    KitchenProprioceptive,
    R3MRewardFunction,
    S3DRewardFunction,
    AttrDict,
    LIVVLMRewardFunction,
    CLIPVLMRewardFunction,
    XCLIPVLMRewardFunction,
    ViLTVLMRewardFunction,
)
from argparse import ArgumentParser
import os

os.environ["TOKENIZERS_PARALLELISM"] = "False"

clip_models = ["openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"]
# xclip_models = ["microsoft/xclip-base-patch16-zero-shot", "microsoft/xclip-base-patch32", "microsoft/xclip-base-patch16"] #"microsoft/xclip-base-patch32-16-frames", "microsoft/xclip-base-patch16-16-frames"]
xclip_models = [
    "microsoft/xclip-base-patch16-zero-shot",
    "microsoft/xclip-base-patch16",
    "microsoft/xclip-large-patch14",
]  # "microsoft/xclip-base-patch32-16-frames", "microsoft/xclip-base-patch16-16-frames"]
vilt_models = [
    "dandelin/vilt-b32-finetuned-coco",
    "dandelin/vilt-b32-finetuned-flickr30k",
]
vlm_choices = ["r3m", "s3d", *clip_models, *xclip_models, *vilt_models]
# vlm_choices = [*xclip_models, *vilt_models]
# vlm_choices = ["r3m", "s3d", *clip_models, ]
# vlm_choices = [*clip_models, *xclip_models, *vilt_models]
# vlm_choices = [*xclip_models, *vilt_models]

results_dir = "skill_ranking_results"
results_list = "skill_ranking_results/results_object_prompts.csv"
results_list_custom_prompts = "skill_ranking_results/results_custom_prompts.csv"
prompts = [
    #      'open the microwave',
    #      'move the pot',
    #      'turn on the stove',
    "arm moving",
    "arm lifting object",
    "arm rotating object",
    "arm dropping object",
    "arm grasping object",
    "arm releasing object",
]


gif_to_skill_map = {
    "demo_gifs/bridge_gifs/": [
        "robot closing microwave",
        "robot opening fridge",
        "robot opening microwave",
        "robot putting knife on cutting board",
        "robot putting pan in sink",
        "robot taking can out of pan",
    ],
    "demo_gifs/franka_demo/": [
        "robot opening cabinet door",
        "robot opening microwave door",
        "robot turning knob",
        "robot switching light on",
    ],
    "demo_gifs/collected_gifs/": [
        "human opening door",
        "human opening microwave",
        "human opening slide door",
        "human turning light on",
    ],
    "demo_gifs/metaworld_demo/": [
        "robot closing green drawer",
        "robot opening black box",
        "robot opening green drawer",
        "robot turning faucet",
    ],
}


def readGif(filename, asNumpy=True):
    """readGif(filename, asNumpy=True)

    Read images from an animated GIF file.  Returns a list of numpy
    arrays, or, if asNumpy is false, a list if PIL images.

    """

    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")

    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")

    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError("File not found: " + str(filename))

    # Load file using PIL
    with PIL.Image.open(filename) as pilIm:
        pilIm.seek(0)
        # Read all images inside
        images = []
        try:
            while True:
                # Get image as numpy array
                tmp = pilIm.convert()  # Make without palette
                a = np.asarray(tmp)
                if len(a.shape) == 0:
                    raise MemoryError("Too little memory to convert PIL image to array")
                # Store, and next
                images.append(a)
                pilIm.seek(pilIm.tell() + 1)
        except EOFError:
            pass

    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for im in images2:
            images.append(PIL.Image.fromarray(im))

    # Done
    return images


def calculate_similarity(img_paths, prompts, vlm_model, name):
    plt.clf()
    similarities = np.zeros((len(img_paths), len(prompts)))
    for i in range(len(img_paths)):
        frames = np.array(readGif(img_paths[i]))
        frames = frames[:128]
        # if np.equal(np.mod(frames, 1).all(), 0):
        #    frames = frames/255
        # frames is T x H x W x C
        # Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1]

        # frames = frames.transpose(0, 4, 1, 2, 3)
        if frames.shape[2] % 2 != 0:
            frames = frames[:, :, :-1, :, :]
        if frames.shape[-1] > 3:
            # convert to rgb from rgba
            frames = frames[..., :3]
        video = torch.from_numpy(frames)
        per_timestep_scores = vlm_model.compute_per_timestep_lang_rewards(
            video.clone(), prompts
        )
        # this is a per-timestep score of shape (num_prompts, num_timesteps)
        # we return a row of the similarity matrix by avging over the timesteps
        per_timestep_scores = per_timestep_scores.mean(dim=1).detach().numpy()
        similarities[i] = per_timestep_scores

    # save the similarity matrix
    sns.set(font_scale=0.6)
    renamed_img_paths = [path.split("/")[-1].split(".")[0] for path in img_paths]
    # make the xticks angled at like 15 degrees
    ax = sns.heatmap(
        similarities,
        annot=True,
        fmt=".2f",
        linewidth=0.5,
        cmap="Blues",
        xticklabels=prompts,
        yticklabels=renamed_img_paths,
    )
    plt.yticks(rotation=45)
    plt.xticks(rotation=15)
    ax.figure.savefig(f"{name}_heat.pdf")
    # get diagonal of the similarity matrix
    diagonal_means = np.diagonal(similarities).mean()
    return diagonal_means


def plot_video(frames, prompts, rewards, interval=60):
    # sns remove grid
    sns.set_style("whitegrid", {"axes.grid": False})

    # First set up the figure, the axis, and the plot element we want to animate
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax1, ax2 = axes

    im = ax1.imshow(frames[0])

    lines = []
    for prompt, reward in zip(prompts, rewards):
        (line,) = ax2.plot(reward, lw=2, label=prompt)
        lines.append(line)
    ax2.legend()
    m = rewards.min()
    M = rewards.max()
    ax2.set_xlim(0, len(frames))
    ax2.set_ylim(m - (M - m) * 0.1, M + (M - m) * 0.1)

    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        im.set_data(frames[i])
        for line, rew in zip(lines, rewards):
            line.set_data(range(i), rew[:i])
        return (im, lines)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(frames), interval=interval, blit=False
    )

    return anim


def create_vlm_video(
    video: torch.Tensor,
    prompts: List[str],
    model: VLMRewardFunction,
    name: str,
    interval: int = 60,
):
    plt.clf()
    per_timestep_lang_rewards = model.compute_per_timestep_lang_rewards(
        video.clone(), prompts
    )
    anim = plot_video(video, prompts, per_timestep_lang_rewards, interval=interval)
    anim.save(f"{name}.mp4")


def construct_model(args):
    device = args.device
    if "r3m" in args.vlm:
        r3m_config = AttrDict()
        r3m_config.model_name = "resnet50"
        r3m_config.vlm = args.vlm
        r3m_config.device = device
        r3m_config.window_size = int(args.vlm.split("-")[-1])
        r3m_config.reward_type = args.reward_type
        return R3MRewardFunction(r3m_config)
    elif args.vlm == "s3d":
        s3d_config = AttrDict()
        s3d_config.device = device
        s3d_config.vlm = args.vlm
        s3d_config.batch_size = 4
        s3d_config.stack_size = 16
        s3d_config.reward_type = args.reward_type
        return S3DRewardFunction(s3d_config)
    elif args.vlm in clip_models:
        return CLIPVLMRewardFunction(args)
    elif args.vlm in xclip_models:
        return XCLIPVLMRewardFunction(args)
    elif args.vlm in vilt_models:
        return ViLTVLMRewardFunction(args)
    elif args.vlm == "LIV":
        return LIVVLMRewardFunction(args)
    elif args.vlm == "state_diff_ablation":
        return StateDiffRewardFunction(args)
    elif args.vlm == "kitchen_proprioceptive":
        return KitchenProprioceptive(args)
    else:
        raise NotImplementedError


def main(rew_func: VLMRewardFunction, config):
    vlm_name = config.vlm.split("/")[-1]
    vlm_path = os.path.join(results_dir, vlm_name)
    os.makedirs(vlm_path, exist_ok=True)
    # FRANKA KICHEN
    videodata = skvideo.io.vread("./other_vids/simpl_task1_epi20.mp4")
    video = torch.as_tensor(videodata)
    create_vlm_video(video, prompts, rew_func, f"{vlm_path}/kitchen")

    # FRANKA KITCHEN MULTIPLE VIEWS
    file_name = "other_vids/FK1_RelaxFixed_v2d-v4_600_20230506-124421_trace.h5"

    trial_num = 600
    with h5py.File(file_name, "r") as f:
        visual_dict = f[f"Trial{trial_num}"]["env_infos"]["visual_dict"]
        for k, v in visual_dict.items():
            if "rgb" in k:
                torch_vid = torch.from_numpy(v[()])
                create_vlm_video(
                    torch_vid, prompts, rew_func, f"{vlm_path}/kitchen_{k}"
                )

    # Karl kitchen data
    videodata = skvideo.io.vread("./other_vids/GX010441_MKBT.MP4")
    video = torch.from_numpy(videodata)[100:800]
    create_vlm_video(video, prompts, rew_func, f"{vlm_path}/karl_kitchen", interval=20)

    # make the heatmaps
    diagonal_similarities = []
    for key in gif_to_skill_map.keys():
        gif_paths = [key + gif for gif in os.listdir(key)]
        key_name = key.split("/")[1]
        gif_paths.sort()
        diagonal_similarities.append(
            calculate_similarity(
                gif_paths, gif_to_skill_map[key], rew_func, f"{vlm_path}/{key_name}"
            )
        )
    # save this to a shared file
    # first check if it exists:
    if os.path.exists(results_list):
        df = pd.read_csv(results_list)
    else:
        df = pd.DataFrame(
            columns=["vlm", "bridge", "franka", "collected", "metaworld", "avg"]
        )
    # if vlm_name in df["vlm"].values:
    #    df = df[df["vlm"] != vlm_name]
    df.loc[len(df)] = [vlm_name, *diagonal_similarities, np.mean(diagonal_similarities)]
    df.to_csv(results_list, index=False)

    # now do the same thing but with custom prompts
    diagonal_similarities = []
    for key in gif_to_skill_map.keys():
        gif_paths = [key + gif for gif in os.listdir(key)]
        key_name = key.split("/")[1]
        gif_paths.sort()
        diagonal_similarities.append(
            calculate_similarity(
                gif_paths, prompts, rew_func, f"{vlm_path}/{key_name}_custom_prompts"
            )
        )
    """
    print(diagonal_similarities)
    # save this to a shared file
    # first check if it exists:
    if os.path.exists(results_list_custom_prompts):
        df = pd.read_csv(results_list_custom_prompts)
    else:
        df = pd.DataFrame(columns=["vlm", "bridge", "franka", "collected", "metaworld", "avg"])
    df.loc[len(df)] = [vlm_name, *diagonal_similarities, np.mean(diagonal_similarities)]
    df.to_csv(results_list_custom_prompts, index=False)
    """


parser = ArgumentParser()
parser.add_argument("--vlm", type=str, choices=[vlm_choices], help="vlm model to use")
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
)
parser.add_argument(
    "--reward-type",  # only used for CLIP right now
    type=str,
    default="diff",
    choices=["diff", "default"],
)
parser.add_argument("--run_all", action="store_true", help="run all vlm models")

if __name__ == "__main__":
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"
    args.device = device
    if args.run_all:
        # tqdm
        for i in tqdm(range(len(vlm_choices))):
            torch.cuda.empty_cache()
            vlm = vlm_choices[i]
            args.vlm = vlm
            if args.vlm == "r3m":
                r3m_config = AttrDict()
                r3m_config.model_name = "resnet50"
                r3m_config.device = device
                main(R3MRewardFunction(r3m_config), args)
            elif args.vlm == "s3d":
                s3d_config = AttrDict()
                s3d_config.device = device
                s3d_config.batch_size = 1
                s3d_config.stack_size = 16
                main(S3DRewardFunction(s3d_config), args)
            elif args.vlm in clip_models:
                main(CLIPVLMRewardFunction(args), args)
            elif args.vlm in xclip_models:
                main(XCLIPVLMRewardFunction(args), args)
            elif args.vlm in vilt_models:
                main(ViLTVLMRewardFunction(args), args)
            else:
                raise NotImplementedError

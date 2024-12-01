# import d4rl
import tqdm
import copy
import h5py
import pyxis as px
import torchvision
import torch
import gym
import numpy as np
import itertools
from torch.utils.data import default_collate
import kornia.augmentation as K

from extract.components.data_loader import Dataset
from extract.rl.envs.kitchen import KitchenEnv
from extract.utils.general_utils import AttrDict
from extract.utils.video_utils import resize_video
from extract.utils.pytorch_utils import RepeatedDataLoader

DEBUG = False


class D4RLSequenceSplitDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(
        self,
        data_dir,
        data_conf,
        phase,
        resolution=None,
        shuffle=True,
        dataset_size=-1,
    ):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.use_image = self.spec.use_image
        self.subseq_len = self.spec.subseq_len
        self.load_all_frames = (
            self.spec.load_all_frames if "load_all_frames" in self.spec else False
        )
        self.n_input_frames = (
            self.spec.n_input_frames
            if "n_input_frames" in self.spec
            else self.subseq_len
        )
        self.use_language = (
            self.spec.use_language if "use_language" in self.spec else False
        )
        if "pad_n_steps" not in self.spec:
            self.spec.pad_n_steps = 0
        self.no_load_in_ram = (
            self.spec.no_load_in_ram if "no_load_in_ram" in self.spec else False
        )
        self.remove_goal = (
            self.spec.remove_goal if "remove_goal" in self.spec else False
        )
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle
        self.image_aug = self.spec.image_aug if "image_aug" in self.spec else False
        if self.image_aug:
            aug_cfg = self.spec.image_aug_specs
            self.image_aug_fn = torch.nn.Sequential(
                K.ColorJitter(
                    aug_cfg.brightness,
                    aug_cfg.contrast,
                    aug_cfg.saturation,
                    aug_cfg.hue,
                    p=aug_cfg.color_p,
                ),
                K.RandomGaussianNoise(std=aug_cfg.noise_std, p=aug_cfg.noise_p),
                K.RandomChannelShuffle(p=aug_cfg.channel_shuffle_p),
                K.RandomAffine(
                    degrees=aug_cfg.degrees,
                    translate=(aug_cfg.translate, aug_cfg.translate),
                ),
                K.RandomErasing(p=aug_cfg.erase_p),
                # K.Normalize(
                #    mean=image_processor.image_mean, std=image_processor.image_std
                # ),
            )
        if self.use_image or self.spec.load_data:
            if "h5" in data_dir or "hdf5" in data_dir:
                self.dataset = h5py.File(data_dir, "r")
                self.is_lmdb = False
            else:
                self.dataset = px.Reader(data_dir, lock=False)
                self.is_lmdb = True
        else:
            env = gym.make(self.spec.env_name)
            self.dataset = env.get_dataset()

        # split dataset into sequences
        start = 0
        self.seqs = []
        obs_key = "rendered_frames" if self.use_image else "observations"
        self.obs_key = obs_key
        self.resolution = resolution
        print("Loading Data")
        if not self.no_load_in_ram or not self.is_lmdb:
            seq_end_idxs = np.where(self.dataset["terminals"])[0]
            if DEBUG:
                seq_end_idxs = seq_end_idxs[:100]
            for end_idx in tqdm.tqdm(seq_end_idxs):
                # if end_idx + 1 - start < self.subseq_len:
                # if end_idx + 1 - start < self.min_subseq_len - self.spec.pad_n_steps:
                if end_idx + 1 - start < self.subseq_len:
                    continue  # skip too short demos
                if self.use_image:
                    states = torch.from_numpy(
                        self.dataset[obs_key][start : end_idx + 1]
                    ).permute(0, 3, 1, 2)
                    states = torchvision.transforms.functional.resize(
                        states,
                        (resolution, resolution),
                        antialias=True,
                    ).numpy()
                else:
                    states = self.dataset[obs_key][start : end_idx + 1]
                # assert np.allclose(
                #    self.dataset["lang_embeds"][start],
                #    self.dataset["lang_embeds"][end_idx],
                # )
                if self.use_language:
                    lang_embeds = self.dataset["lang_embeds"][
                        start : end_idx + 1
                    ].astype("float32")
                else:
                    lang_embeds = None

                self.seqs.append(
                    AttrDict(
                        states=states,
                        actions=self.dataset["actions"][start : end_idx + 1],
                        lang=lang_embeds,
                    )
                )
                start = end_idx + 1

            # 0-pad sequences for skill-conditioned training
            if "pad_n_steps" in self.spec and self.spec.pad_n_steps > 0:
                for seq in self.seqs:
                    for key, value in seq.items():
                        if key == "states":
                            # repeat the frame first frame
                            seq.states = np.concatenate(
                                (
                                    np.repeat(value[:1], self.spec.pad_n_steps, axis=0),
                                    value,
                                )
                            )
                        elif value is None:
                            continue
                        elif len(value.shape) > 1:
                            seq[key] = np.concatenate(
                                (
                                    np.zeros(
                                        (self.spec.pad_n_steps, value.shape[1]),
                                        dtype=value.dtype,
                                    ),
                                    value,
                                )
                            )
                        else:
                            seq[key] = np.concatenate(
                                (
                                    np.zeros(
                                        (self.spec.pad_n_steps,),
                                        dtype=value.dtype,
                                    ),
                                    value,
                                )
                            )

            # filter demonstration sequences
            if "filter_indices" in self.spec:
                print(
                    "!!! Filtering kitchen demos in range {} !!!".format(
                        self.spec.filter_indices
                    )
                )
                if not isinstance(self.spec.filter_indices[0], list):
                    self.spec.filter_indices = [self.spec.filter_indices]
                self.seqs = list(
                    itertools.chain.from_iterable(
                        [
                            list(
                                itertools.chain.from_iterable(
                                    itertools.repeat(x, self.spec.demo_repeats)
                                    for x in self.seqs[fi[0] : fi[1] + 1]
                                )
                            )
                            for fi in self.spec.filter_indices
                        ]
                    )
                )
                import random

                random.shuffle(self.seqs)

            self.n_seqs = len(self.seqs)
            if self.phase == "train":
                self.start = 0
                self.end = int(self.SPLIT.train * self.n_seqs)
            elif self.phase == "val":
                self.start = int(self.SPLIT.train * self.n_seqs)
                self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            else:
                self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
                self.end = self.n_seqs
        else:
            print("Data not being loaded from and shuffled in ram, stored on disk")
            # seq_end_idxs = np.where(self.dataset["terminals"])[0]
            seq_end_idxs = np.where(self.grab_data_from_dataset("terminals", 0, None))[
                0
            ]
            self.sequence_starts = []
            self.sequence_ends = []
            for end_idx in seq_end_idxs:
                if end_idx + 1 - start < self.subseq_len:
                    continue  # skip too short demos
                self.sequence_starts.append(start)
                self.sequence_ends.append(end_idx)
                start = end_idx + 1

    def grab_data_from_dataset(self, key, start, end):
        if self.is_lmdb:
            # grab data from lmdb dataset
            if end is None:
                end = len(self.dataset)
            out = self.dataset[start:end]
            out = np.stack([x[key] for x in out], axis=0)
        else:
            # grab data from hdf5 dataset
            out = self.dataset[key][start:end]
        return out

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        if self.no_load_in_ram:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        output = AttrDict(
            states=seq.states[start_idx : start_idx + self.subseq_len],
            actions=seq.actions[start_idx : start_idx + self.subseq_len - 1],
            pad_mask=np.ones((self.subseq_len,), dtype=np.float32),
        )
        if self.use_language:
            output.lang = seq.lang[
                start_idx : start_idx + self.subseq_len - self.n_input_frames
            ]
        if self.remove_goal:
            output.states = output.states[..., : int(output.states.shape[-1] / 2)]
        if self.use_image:
            output.images = self._preprocess_images(output.states)
            output.pop("states")
        return output

    @property
    def n_states(self):
        if self.is_lmdb:
            return len(self.dataset)
        else:
            return self.dataset[self.obs_key].shape[0]

    def _sample_seq(self):
        if self.no_load_in_ram:
            # load from disk dataset
            # valid_seq = False
            # while not valid_seq:
            #    start_idx = np.random.randint(0, self.n_states - self.subseq_len + 1)
            #    end_idx = start_idx + self.subseq_len
            #    valid_seq = not np.any(
            #        self.grab_data_from_dataset("terminals", start_idx, end_idx - 1)
            #    )
            rand_idx = np.random.randint(0, len(self.sequence_starts))
            seq_start_idx = self.sequence_starts[rand_idx]
            seq_end_idx = self.sequence_ends[rand_idx]

            if hasattr(self, "min_subseq_len") and hasattr(self, "max_subseq_len"):
                # for the variable length skills in the cluster model
                start_idx = np.random.randint(
                    seq_start_idx, seq_end_idx - self.min_subseq_len + 1
                )
                end_idx = min(start_idx + self.max_subseq_len, seq_end_idx)
            else:
                start_idx = np.random.randint(
                    seq_start_idx, seq_end_idx - self.subseq_len + 1
                )
                end_idx = start_idx + self.subseq_len

            if self.use_image:
                img_end_idx = (
                    end_idx if self.load_all_frames else start_idx + self.n_input_frames
                )
                states = np.transpose(
                    self.grab_data_from_dataset(self.obs_key, start_idx, img_end_idx),
                    [0, 3, 1, 2],
                )
            else:
                states = self.grab_data_from_dataset(self.obs_key, start_idx, end_idx)
            if self.use_language:
                lang_embeds = self.grab_data_from_dataset(
                    "lang_embeds", start_idx, end_idx
                ).astype("float32")
            else:
                lang_embeds = None
            skill_progress = np.linspace(0, 1, seq_end_idx - seq_start_idx)
            if hasattr(self, "cluster_logprobs"):
                cluster_logprobs = self.cluster_logprobs[start_idx:end_idx]
                cluster_assignments = self.cluster_assignments[start_idx:end_idx]
            else:
                cluster_logprobs = None
                cluster_assignments = None
            return AttrDict(
                states=states,
                actions=self.grab_data_from_dataset("actions", start_idx, end_idx),
                lang=lang_embeds,
                skill_logprobs=cluster_logprobs,
                skills=cluster_assignments,
                skill_progress=skill_progress,
            )
        return np.random.choice(self.seqs[self.start : self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.n_states / self.subseq_len)

    def _preprocess_images(self, images):
        assert images.dtype == np.uint8, "image need to be uint8!"
        # don't need the resizing and transposing because we already resize with torchvision
        # images = resize_video(images, (self.resolution, self.resolution))
        # images = np.transpose(images, [0, 3, 1, 2])  # convert to channel-first
        images = images.astype(np.float32) / 255 * 2 - 1
        assert images.dtype == np.float32, "image need to be float32!"
        return images

    def get_data_loader(self, batch_size, n_repeat):
        print("len {} dataset {}".format(self.phase, len(self)))
        assert self.device in ["cuda", "cpu"]  # Otherwise the logic below is wrong
        return RepeatedDataLoader(
            self,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_worker if not DEBUG else 0,
            drop_last=True,
            n_repeat=n_repeat,
            pin_memory=self.device == "cuda",
            worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        collated = default_collate(batch)
        # if loaded in ram we resized, otherwise we need to resize on the fly
        if self.use_image and (self.image_aug or self.no_load_in_ram):
            images = collated.images
            bs = images.shape[0]
            length = images.shape[1]
            if images.shape[2] != self.resolution:
                images = torchvision.transforms.functional.resize(
                    images.reshape(-1, *images.shape[2:]),
                    (self.resolution, self.resolution),
                    antialias=True,
                )
            if self.image_aug:
                # convert back to [0, 1] for Kornia
                images = (images + 1) / 2
                images = (
                    self.image_aug_fn(images) * 2 - 1
                )  # aug and then convert back to [-1, 1]
            images = images.reshape(bs, length, *images.shape[1:])
            collated.images = images
        return collated


class RLPDD4RLSequenceSplitDataset(D4RLSequenceSplitDataset):
    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        if self.no_load_in_ram:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        end_index = start_idx + self.subseq_len
        output = AttrDict(
            states=seq.states[start_idx : start_idx + self.n_input_frames],
            next_states=seq.states[end_index - self.n_input_frames + 1 : end_index + 1],
            # done = # TODO,
            # reward = # TODO,
            actions=seq.actions[start_idx : start_idx + self.subseq_len - 1],
            pad_mask=np.ones((self.subseq_len,), dtype=np.float32),
        )
        if self.use_language:
            output.lang = seq.lang[
                start_idx : start_idx + self.subseq_len - self.n_input_frames
            ]
        if self.remove_goal:
            output.states = output.states[..., : int(output.states.shape[-1] / 2)]
        if self.use_image:
            output.images = self._preprocess_images(output.states)
            output.pop("states")
        return output

class SkillClusterD4RLSequenceSplitDataset(D4RLSequenceSplitDataset):
    def __init__(
        self,
        data_dir,
        data_conf,
        phase,
        resolution=None,
        shuffle=True,
        dataset_size=-1,
    ):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.use_image = self.spec.use_image
        self.max_subseq_len = self.spec.max_subseq_len
        self.n_input_frames = self.spec.n_input_frames
        self.load_all_frames = (
            self.spec.load_all_frames if "load_all_frames" in self.spec else False
        )
        self.min_subseq_len = (
            self.spec.min_subseq_len
        )  # TODO: add min_subseq_len in the config
        self.remove_goal = (
            self.spec.remove_goal if "remove_goal" in self.spec else False
        )
        self.use_language = (
            self.spec.use_language if "use_language" in self.spec else False
        )
        self.dataset_size = dataset_size
        self.start_skill_anywhere = self.spec.start_skill_anywhere
        self.no_load_in_ram = (
            self.spec.no_load_in_ram if "no_load_in_ram" in self.spec else False
        )
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle
        self.image_aug = self.spec.image_aug if "image_aug" in self.spec else False
        if self.image_aug:
            aug_cfg = self.spec.image_aug_specs
            self.image_aug_fn = torch.nn.Sequential(
                K.ColorJitter(
                    aug_cfg.brightness,
                    aug_cfg.contrast,
                    aug_cfg.saturation,
                    aug_cfg.hue,
                    p=aug_cfg.color_p,
                ),
                K.RandomGaussianNoise(std=aug_cfg.noise_std, p=aug_cfg.noise_p),
                K.RandomChannelShuffle(p=aug_cfg.channel_shuffle_p),
                K.RandomAffine(
                    degrees=aug_cfg.degrees,
                    translate=(aug_cfg.translate, aug_cfg.translate),
                ),
                K.RandomErasing(p=aug_cfg.erase_p),
                # K.Normalize(
                #    mean=image_processor.image_mean, std=image_processor.image_std
                # ),
            )
        if "h5" in data_dir or "hdf5" in data_dir:
            self.dataset = h5py.File(data_dir, "r")
            self.is_lmdb = False
        else:
            self.dataset = px.Reader(data_dir, lock=False)
            self.is_lmdb = True
        with h5py.File(self.spec.cluster_data_file, "r") as f:
            cluster_assignments = f["clusters"][()]
            cluster_logprobs = f["logprobs"][()]
            if len(cluster_logprobs.shape) < 2:
                use_logprobs = False
                print("Logprobs is of wrong shape")
            else:
                use_logprobs = True
                cluster_logprobs = cluster_logprobs.transpose(1, 0)
        start = 0
        self.seqs = []
        obs_key = "rendered_frames" if self.use_image else "observations"
        self.obs_key = obs_key
        self.resolution = resolution
        skill_lengths = []
        skipped_traj_counter = 0
        if not self.no_load_in_ram or not self.is_lmdb:
            print("Loading Data")
            # split dataset into sequences
            seq_end_idxs = np.where(self.dataset["terminals"])[0]
            # detect changes in skill
            skill_end_idxs = np.where(
                (cluster_assignments[1:] - cluster_assignments[:-1]) != 0
            )[0]
            # now combine with the seq_end_idxs
            seq_end_idxs = np.concatenate((seq_end_idxs, skill_end_idxs))
            # remove duplicates and sort
            seq_end_idxs = np.unique(seq_end_idxs)
            if DEBUG:
                seq_end_idxs = seq_end_idxs[:300]
            for end_idx in tqdm.tqdm(seq_end_idxs):
                # if end_idx + 1 - start < self.min_subseq_len - self.spec.pad_n_steps:
                if end_idx + 1 - start < self.min_subseq_len:
                    skipped_traj_counter += 1
                    # start = end_idx + 1
                    continue  # skip too short demos
                if self.use_image:
                    states = torch.from_numpy(
                        self.dataset[obs_key][start : end_idx + 1]
                    ).permute(0, 3, 1, 2)
                    states = torchvision.transforms.functional.resize(
                        states,
                        (resolution, resolution),
                        antialias=True,
                    ).numpy()
                else:
                    states = self.dataset[obs_key][start : end_idx + 1]
                skills = cluster_assignments[start : end_idx + 1]
                if use_logprobs:
                    skill_logprobs = cluster_logprobs[start : end_idx + 1]
                else:
                    skill_logprobs = np.zeros((end_idx + 1 - start, 1))
                # fold skill assignment into next
                # skills[:] = skills[-1]
                actions = self.dataset["actions"][start : end_idx + 1]
                if len(actions) > 1:
                    skill_progress = np.linspace(0, 1, len(actions))
                else:
                    skill_progress = np.array([1.0])
                if self.use_language:
                    lang_embeds = self.dataset["lang_embeds"][
                        start : end_idx + 1
                    ].astype("float32")
                else:
                    lang_embeds = None
                skill_lengths.append(len(actions))
                self.seqs.append(
                    AttrDict(
                        states=states,
                        actions=actions,
                        skills=skills,
                        skill_logprobs=skill_logprobs,
                        skill_progress=skill_progress,
                        lang=lang_embeds,
                    )
                )
                start = end_idx + 1

            # pad sequences for frame-stack training since we have defined skill starts and ends
            if "pad_n_steps" in self.spec and self.spec.pad_n_steps > 0:
                for seq in self.seqs:
                    for key, value in seq.items():
                        if key == "states":
                            # repeat the frame first frame
                            seq.states = np.concatenate(
                                (
                                    np.repeat(value[:1], self.spec.pad_n_steps, axis=0),
                                    value,
                                )
                            )
                        elif value is None:
                            continue
                        elif len(value.shape) > 1:
                            seq[key] = np.concatenate(
                                (
                                    np.zeros(
                                        (self.spec.pad_n_steps, value.shape[1]),
                                        dtype=value.dtype,
                                    ),
                                    value,
                                )
                            )
                        else:
                            seq[key] = np.concatenate(
                                (
                                    np.zeros(
                                        (self.spec.pad_n_steps,),
                                        dtype=value.dtype,
                                    ),
                                    value,
                                )
                            )

            # filter demonstration sequences
            if "filter_indices" in self.spec:
                print(
                    "!!! Filtering kitchen demos in range {} !!!".format(
                        self.spec.filter_indices
                    )
                )
                if not isinstance(self.spec.filter_indices[0], list):
                    self.spec.filter_indices = [self.spec.filter_indices]
                self.seqs = list(
                    itertools.chain.from_iterable(
                        [
                            list(
                                itertools.chain.from_iterable(
                                    itertools.repeat(x, self.spec.demo_repeats)
                                    for x in self.seqs[fi[0] : fi[1] + 1]
                                )
                            )
                            for fi in self.spec.filter_indices
                        ]
                    )
                )
                import random

                random.shuffle(self.seqs)

            self.n_seqs = len(self.seqs)

            if self.phase == "train":
                self.start = 0
                self.end = int(self.SPLIT.train * self.n_seqs)
            elif self.phase == "val":
                self.start = int(self.SPLIT.train * self.n_seqs)
                self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            else:
                self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
                self.end = self.n_seqs

            print(f"Average skill length: {np.mean(skill_lengths)}")
            print(f"Skipped {skipped_traj_counter} trajectories")
        else:
            print("Data not being loaded from and shuffled in ram, stored on disk")
            seq_end_idxs = np.where(self.grab_data_from_dataset("terminals", 0, None))[
                0
            ]
            # need to save the cluster assignments and logprobs as well to load in _sample_seq
            self.cluster_assignments = cluster_assignments
            self.cluster_logprobs = cluster_logprobs
            self.sequence_starts = []
            self.sequence_ends = []
            for end_idx in seq_end_idxs:
                if end_idx + 1 - start < self.min_subseq_len:
                    continue  # skip too short demos
                self.sequence_starts.append(start)
                self.sequence_ends.append(end_idx)
                start = end_idx + 1

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        if self.start_skill_anywhere:
            start_idx = np.random.randint(0, seq.actions.shape[0])
        else:
            if self.max_subseq_len >= seq.actions.shape[0]:
                start_idx = 0
            else:
                start_idx = np.random.randint(
                    0, seq.actions.shape[0] - self.max_subseq_len + 1
                )
        states = seq.states[start_idx : start_idx + self.max_subseq_len]
        actions = seq.actions[start_idx : start_idx + self.max_subseq_len]
        progress = seq.skill_progress[start_idx : start_idx + self.max_subseq_len]
        last_action_index = np.array([actions.shape[0] - 1])
        skill_logprobs = seq.skill_logprobs[start_idx : start_idx + self.max_subseq_len]
        # skill_logprobs is a sequence of logprobs for each skill, we convert it to a single number
        # by getting logprobs mean over each timestep
        skill_logprobs = np.mean(skill_logprobs, axis=0)
        output = AttrDict(
            states=states,
            actions=actions,
            skills=seq.skills[start_idx : start_idx + self.max_subseq_len],
            skill_logprobs=skill_logprobs,
            skill_progress=progress,
            action_pad_mask=np.ones((actions.shape[0],), dtype=np.float32),
            last_action_index=last_action_index,
        )
        if self.use_language:
            output.lang = seq.lang[start_idx : start_idx + self.max_subseq_len]
        if self.remove_goal:
            output.states = output.states[..., : int(output.states.shape[-1] / 2)]
        if self.use_image:
            output.images = self._preprocess_images(output.states)
            output.pop("states")
        return output

    def get_data_loader(self, batch_size, n_repeat):
        print("len {} dataset {}".format(self.phase, len(self)))
        assert self.device in ["cuda", "cpu"]  # Otherwise the logic below is wrong
        return RepeatedDataLoader(
            self,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_worker if not DEBUG else 0,
            drop_last=True,
            n_repeat=n_repeat,
            pin_memory=self.device == "cuda",
            worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x),
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        # collates variable length inputs by padding them together
        (
            states,
            actions,
            skills,
            skill_progress,
            action_pad_mask,
            last_action_index,
            skill_logprobs,
            lang,
        ) = ([], [], [], [], [], [], [], [])
        for b in batch:
            if self.use_image:
                states.append(torch.from_numpy(b.images))
            else:
                states.append(torch.from_numpy(b.states))
            actions.append(torch.from_numpy(b.actions))
            skills.append(torch.from_numpy(b.skills))
            skill_progress.append(torch.from_numpy(b.skill_progress))
            action_pad_mask.append(torch.from_numpy(b.action_pad_mask))
            last_action_index.append(torch.from_numpy(b.last_action_index))
            skill_logprobs.append(torch.from_numpy(b.skill_logprobs))
            if self.use_language:
                lang.append(torch.from_numpy(b.lang))
        states = torch.nn.utils.rnn.pad_sequence(states, batch_first=True)
        # if loaded in ram we resized, otherwise we need to resize on the fly
        if self.use_image and (self.image_aug or self.no_load_in_ram):
            bs = states.shape[0]
            length = states.shape[1]
            if states.shape[2] != self.resolution:
                states = torchvision.transforms.functional.resize(
                    states.reshape(-1, *states.shape[2:]),
                    (self.resolution, self.resolution),
                    antialias=True,
                )
            if self.image_aug:
                # convert back to [0, 1] for Kornia
                states = (states + 1) / 2
                states = (
                    self.image_aug_fn(states) * 2 - 1
                )  # aug and then convert back to [-1, 1]
            states = states.reshape(bs, length, *states.shape[1:])
        if self.use_language:
            lang = torch.nn.utils.rnn.pad_sequence(lang, batch_first=True)
        actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True)
        skills = torch.nn.utils.rnn.pad_sequence(skills, batch_first=True)
        skill_logprobs = torch.nn.utils.rnn.pad_sequence(
            skill_logprobs, batch_first=True
        ).float()
        skill_progress = torch.nn.utils.rnn.pad_sequence(
            skill_progress, batch_first=True
        ).float()
        action_pad_mask = torch.nn.utils.rnn.pad_sequence(
            action_pad_mask, batch_first=True
        ).float()
        last_action_index = torch.nn.utils.rnn.pad_sequence(
            last_action_index, batch_first=True
        ).squeeze(-1)
        ret = AttrDict(
            images=states,
            states=states,
            actions=actions,
            skills=skills,
            skill_progress=skill_progress,
            skill_logprobs=skill_logprobs,
            action_pad_mask=action_pad_mask,
            last_action_index=last_action_index.long(),
        )
        if self.use_language:
            ret.lang = lang
        return ret

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.n_states / self.max_subseq_len)


class UVDD4RLDataset(SkillClusterD4RLSequenceSplitDataset):
    """
    No clustering, just using the trajectories segmented by UVD.
    Therefore, always return 0 cluster, and can use with n_clusters=1 argument.
    """

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        if self.start_skill_anywhere:
            start_idx = np.random.randint(0, seq.actions.shape[0])
        else:
            if self.max_subseq_len >= seq.actions.shape[0]:
                start_idx = 0
            else:
                start_idx = np.random.randint(
                    0, seq.actions.shape[0] - self.max_subseq_len + 1
                )
        states = seq.states[start_idx : start_idx + self.max_subseq_len]
        actions = seq.actions[start_idx : start_idx + self.max_subseq_len]
        progress = seq.skill_progress[start_idx : start_idx + self.max_subseq_len]
        last_action_index = np.array([actions.shape[0] - 1])
        # skill_logprobs = seq.skill_logprobs[start_idx : start_idx + self.max_subseq_len]
        # skill_logprobs is a sequence of logprobs for each skill, we convert it to a single number
        # by getting logprobs mean over each timestep
        # skill_logprobs = np.mean(skill_logprobs, axis=0)
        output = AttrDict(
            states=states,
            actions=actions,
            skills=np.zeros((actions.shape[0],), dtype=np.int64),
            skill_progress=progress,
            skill_logprobs=np.empty(1),
            action_pad_mask=np.ones((actions.shape[0],), dtype=np.float32),
            last_action_index=last_action_index,
        )
        if self.use_language:
            output.lang = seq.lang[start_idx : start_idx + self.max_subseq_len]
        if self.remove_goal:
            output.states = output.states[..., : int(output.states.shape[-1] / 2)]
        if self.use_image:
            output.images = self._preprocess_images(output.states)
            output.pop("states")
        return output


class RLPDSkillClusterD4RLSequenceSplitDataset(SkillClusterD4RLSequenceSplitDataset):
    def __getitem__(self, index):
        # TODO: properly load next state
        seq = self._sample_seq()
        if self.start_skill_anywhere:
            start_idx = np.random.randint(0, seq.actions.shape[0])
        else:
            if self.max_subseq_len >= seq.actions.shape[0]:
                start_idx = 0
            else:
                start_idx = np.random.randint(
                    0, seq.actions.shape[0] - self.max_subseq_len + 1
                )
        states = seq.states[start_idx : start_idx + self.max_subseq_len]
        actions = seq.actions[start_idx : start_idx + self.max_subseq_len]
        progress = seq.skill_progress[start_idx : start_idx + self.max_subseq_len]
        last_action_index = np.array([actions.shape[0] - 1])
        skill_logprobs = seq.skill_logprobs[start_idx : start_idx + self.max_subseq_len]
        # skill_logprobs is a sequence of logprobs for each skill, we convert it to a single number
        # by getting logprobs mean over each timestep
        skill_logprobs = np.mean(skill_logprobs, axis=0)
        output = AttrDict(
            states=states,
            actions=actions,
            skills=seq.skills[start_idx : start_idx + self.max_subseq_len],
            skill_logprobs=skill_logprobs,
            skill_progress=progress,
            action_pad_mask=np.ones((actions.shape[0],), dtype=np.float32),
            last_action_index=last_action_index,
        )
        if self.use_language:
            output.lang = seq.lang[start_idx : start_idx + self.max_subseq_len]
        if self.remove_goal:
            output.states = output.states[..., : int(output.states.shape[-1] / 2)]
        if self.use_image:
            output.images = self._preprocess_images(output.states)
            output.pop("states")
        return output

    
    def _collate_fn(self, batch):
        # TODO: collate the additional keys
        (
            states,
            actions,
            skills,
            skill_progress,
            action_pad_mask,
            last_action_index,
            skill_logprobs,
            lang,
        ) = ([], [], [], [], [], [], [], [])
        for b in batch:
            if self.use_image:
                states.append(torch.from_numpy(b.images))
            else:
                states.append(torch.from_numpy(b.states))
            actions.append(torch.from_numpy(b.actions))
            skills.append(torch.from_numpy(b.skills))
            skill_progress.append(torch.from_numpy(b.skill_progress))
            action_pad_mask.append(torch.from_numpy(b.action_pad_mask))
            last_action_index.append(torch.from_numpy(b.last_action_index))
            skill_logprobs.append(torch.from_numpy(b.skill_logprobs))
            if self.use_language:
                lang.append(torch.from_numpy(b.lang))
        states = torch.nn.utils.rnn.pad_sequence(states, batch_first=True)
        # if loaded in ram we resized, otherwise we need to resize on the fly
        if self.use_image and (self.image_aug or self.no_load_in_ram):
            bs = states.shape[0]
            length = states.shape[1]
            if states.shape[2] != self.resolution:
                states = torchvision.transforms.functional.resize(
                    states.reshape(-1, *states.shape[2:]),
                    (self.resolution, self.resolution),
                    antialias=True,
                )
            if self.image_aug:
                # convert back to [0, 1] for Kornia
                states = (states + 1) / 2
                states = (
                    self.image_aug_fn(states) * 2 - 1
                )  # aug and then convert back to [-1, 1]
            states = states.reshape(bs, length, *states.shape[1:])
        if self.use_language:
            lang = torch.nn.utils.rnn.pad_sequence(lang, batch_first=True)
        actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True)
        skills = torch.nn.utils.rnn.pad_sequence(skills, batch_first=True)
        skill_logprobs = torch.nn.utils.rnn.pad_sequence(
            skill_logprobs, batch_first=True
        ).float()
        skill_progress = torch.nn.utils.rnn.pad_sequence(
            skill_progress, batch_first=True
        ).float()
        action_pad_mask = torch.nn.utils.rnn.pad_sequence(
            action_pad_mask, batch_first=True
        ).float()
        last_action_index = torch.nn.utils.rnn.pad_sequence(
            last_action_index, batch_first=True
        ).squeeze(-1)
        ret = AttrDict(
            images=states,
            states=states,
            actions=actions,
            skills=skills,
            skill_progress=skill_progress,
            skill_logprobs=skill_logprobs,
            action_pad_mask=action_pad_mask,
            last_action_index=last_action_index.long(),
        )
        if self.use_language:
            ret.lang = lang
        return ret

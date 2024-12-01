import torch
# from s3dg import S3D
import torch.nn as nn
from r3m import load_r3m
import os
from os.path import expanduser
import omegaconf
import hydra
import gdown
import torch
import torchvision.transforms as T
import numpy as np
import copy
#from liv import load_liv
from collections import UserDict
from transformers import (
    CLIPProcessor,
    CLIPModel,
    XCLIPProcessor,
    XCLIPModel,
    ViltProcessor,
    ViltForImageAndTextRetrieval,
)
import tqdm

BATCH_SIZE = 256


class AttrDict(UserDict):
    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        if key == "data":
            return super().__setattr__(key, value)
        return self.__setitem__(key, value)


class VLMRewardFunction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reward_type = config.reward_type

    def preprocess_frames(self, frames):
        # do normalization and perhaps other stuff here
        raise NotImplementedError

    def preprocess_lang(self, lang):
        # do normalization and embed the lang or something
        raise NotImplementedError

    def compute_video_embed(self, frames):
        raise NotImplementedError

    def compute_lang_reward(self, frames, lang):
        raise NotImplementedError

    def compute_video_reward(self, robot_frames, other_frames):
        raise NotImplementedError

    def compute_per_timestep_lang_rewards(self, frames, lang):
        raise NotImplementedError

    def postprocess_per_timestep_rewards_to_delta(self, probs):
        # expects probs to be something like num sentences x time
        return probs - probs[:, 0]

    def _batch_preprocess_frames(self, frames):
        raise NotImplementedError

    def batch_preprocess_frames(self, frames):
        all_frames = []
        for i in tqdm.trange(0, len(frames), BATCH_SIZE):
            processed_chunk = self._batch_preprocess_frames(
                frames[i : i + BATCH_SIZE].clone()
            )
            all_frames.append(processed_chunk.cpu())
        return torch.cat(all_frames)


def cleanup_config(cfg, device):
    config = copy.deepcopy(cfg)
    keys = config.agent.keys()
    for key in list(keys):
        if key not in VALID_ARGS:
            del config.agent[key]
    config.agent["_target_"] = "r3m.R3M"
    config["device"] = device

    ## Hardcodes to remove the language head
    ## Assumes downstream use is as visual representation
    config.agent["langweight"] = 1.0
    return config.agent


VALID_ARGS = [
    "_target_",
    "device",
    "lr",
    "hidden_dim",
    "size",
    "l2weight",
    "l1weight",
    "langweight",
    "tcnweight",
    "l2dist",
    "bs",
]

class StateDiffRewardFunction(VLMRewardFunction):
    def compute_video_embed(self, states):
        assert len(states.shape) == 2
        return states[1:] - states[:1]


class KitchenProprioceptive(VLMRewardFunction):
    def compute_video_embed(self, states):
        assert len(states.shape) == 2
        return states[1:, :18] - states[:1, :18]


class R3MRewardFunction(VLMRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        home = os.path.join(expanduser("~"), ".r3m")
        modelid = config.model_name
        self.window_size = config.window_size
        if modelid == "resnet50":
            foldername = "r3m_50"
            modelurl = (
                "https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA"
            )
            configurl = (
                "https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8"
            )
        elif modelid == "resnet34":
            foldername = "r3m_34"
            modelurl = (
                "https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE"
            )
            configurl = (
                "https://drive.google.com/uc?id=1RY0NS-Tl4G7M1Ik_lOym0b5VIBxX9dqW"
            )
        elif modelid == "resnet18":
            foldername = "r3m_18"
            modelurl = (
                "https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-"
            )
            configurl = (
                "https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6"
            )
        else:
            raise NameError("Invalid Model ID")

        if not os.path.exists(os.path.join(home, foldername)):
            os.makedirs(os.path.join(home, foldername))
        modelpath = os.path.join(home, foldername, "model.pt")
        configpath = os.path.join(home, foldername, "config.yaml")
        if not os.path.exists(modelpath):
            gdown.download(modelurl, modelpath, quiet=False)
            gdown.download(configurl, configpath, quiet=False)

        modelcfg = omegaconf.OmegaConf.load(configpath)
        cleancfg = cleanup_config(modelcfg, config.device)
        rep = hydra.utils.instantiate(cleancfg)
        rep = torch.nn.DataParallel(rep)
        r3m_state_dict = torch.load(
            modelpath, map_location=torch.device(config.device)
        )["r3m"]
        rep.load_state_dict(r3m_state_dict)
        rep.eval()
        self.model = rep.module
        # self.np_preprocessing_transform = T.Compose(
        #    [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
        # )  # ToTensor() divides by 255
        # self.preprocessing_transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
        self.device = config.device
        self.model = self.model.to(self.device)

    def _batch_preprocess_frames(self, frames):
        processed_frames = frames.permute((0, 3, 1, 2))
        # processed_frames = self.preprocessing_transform(frames)
        if torch.all(processed_frames <= 1):
            processed_frames *= 255
        processed_frames = self.model(processed_frames.to(self.device))
        return processed_frames

    def preprocess_lang(self, lang):
        # no need to embed lang here
        le = self.model.lang_enc(lang)
        return le

    def compute_lang_reward(self, frames, lang):
        # use r3m
        assert frames.shape[0] == 2
        init, curr = frames[0:1], frames[-1:]
        return self.model.lang_rew(init, curr, lang)[0].unsqueeze(-1)

    def compute_video_reward(self, robot_frames, other_frames):
        raise NotImplementedError

    def compute_video_embed(self, frames):
        with torch.no_grad():
            image_embeds = self.batch_preprocess_frames(frames).cpu()
        if self.reward_type == "real_diff":
            return image_embeds[1:] - image_embeds[:-1]
        elif self.reward_type == "first_diff":
            return image_embeds[1:] - image_embeds[:1].repeat(
                (image_embeds.shape[0] - 1, 1)
            )
        else:
            return image_embeds

    def compute_per_timestep_lang_rewards(self, frames, lang):
        with torch.no_grad():
            all_first_frame_features = self.batch_preprocess_frames(frames[:1]).repeat(
                (frames.shape[0], 1)
            )
            per_timestep_frame_features = self.batch_preprocess_frames(frames)
            lang_embeds = self.preprocess_lang(lang)
            # 1 forward pass per language embedding
            rews = []
            for i in range(lang_embeds.shape[0]):
                repeat_lang = lang_embeds[i : i + 1].repeat((frames.shape[0], 1))
                rew = self.model.lang_rew(
                    all_first_frame_features, per_timestep_frame_features, repeat_lang
                )[0]
                rews.append(rew.cpu())
            return torch.softmax(torch.stack(rews), dim=0)


class S3DRewardFunction(VLMRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        # Instantiate the model
        net = S3D("s3d_dict.npy", 512)
        self.device = config.device
        net.to(config.device)

        # # Load the model weights
        net.load_state_dict(torch.load("s3d_howto100m.pth"))

        self.model = net
        self.batch_size = config.batch_size
        self.stack_size = config.stack_size
        self.preprocessing_transform = T.Compose([T.Resize(256), T.CenterCrop(224)])

    def preprocess_lang(self, lang):
        embeds = self.model.text_module(lang)["text_embedding"]
        return embeds

    def _batch_preprocess_frames(self, frames):
        # permute and normalize to 0, 1
        all_preprocessed_frames = []
        frames_stack = []
        original_lens = []
        # for i in range(self.stack_size, len(frames) + self.stack_size, self.stack_size):
        for i in range(0, len(frames)):
            # do a vid per timestep
            # vid = frames[i - self.stack_size : i].permute(0, 3, 1, 2)
            vid = frames[max(i - self.stack_size + 1, 0) : i + 1].permute(0, 3, 1, 2)
            original_lens.append(vid.shape[0])
            if vid.shape[0] < self.stack_size:
                vid = torch.cat(
                    [
                        vid[0]
                        .unsqueeze(0)
                        .repeat(self.stack_size - vid.shape[0], 1, 1, 1),
                        vid,
                    ],
                    dim=0,
                )
            frames_stack.append(vid)
            assert vid.max() > 1
        frames_stack = torch.stack(frames_stack)
        frames_stack = frames_stack / 255.0
        for i in range(0, len(frames_stack), self.batch_size):
            video_stack_chunk = (
                frames_stack[i : i + self.batch_size]
                .permute(0, 2, 1, 3, 4)
                .to(self.device)
            )
            preprocessed_frames = self.model(video_stack_chunk)["video_embedding"].cpu()
            all_preprocessed_frames.append(preprocessed_frames)

        all_preprocessed_frames = torch.cat(all_preprocessed_frames)
        return all_preprocessed_frames, original_lens

    def compute_video_embed(self, frames):
        with torch.no_grad():
            video_embeds, _ = self.batch_preprocess_frames(frames)
        if self.reward_type == "real_diff":
            return video_embeds[1:] - video_embeds[:-1]
        elif self.reward_type == "first_diff":
            return video_embeds[1:] - video_embeds[:1].repeat(
                (video_embeds.shape[0] - 1, 1)
            )
        return video_embeds

    def compute_per_timestep_lang_rewards(self, frames, lang):
        with torch.no_grad():
            video_embeds, original_lens = self.batch_preprocess_frames(frames)
            lang_embeds = self.preprocess_lang(lang).cpu()
            # make empty set of 0's first
            rewards = (video_embeds @ lang_embeds.T).squeeze(1).cpu().T
            # first_rews = torch.zeros(self.stack_size).unsqueeze(0).repeat(rewards.shape[0], 1)
            # rewards = torch.cat((first_rews, rewards), dim=1)
            timestep_aligned_rews = []
            for i in range(rewards.shape[1]):
                length = original_lens[i]
                timestep_aligned_rews.append(rewards[:, i : i + 1].repeat(1, length))
            rewards = torch.cat(timestep_aligned_rews, dim=1)
        return torch.softmax(rewards, dim=0)


class LIVVLMRewardFunction(VLMRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.device = config.device
        self.model = load_liv()
        self.model.eval()

    def _batch_preprocess_frames(self, frames):
        processed_frames = frames.permute((0, 3, 1, 2))
        # processed_frames = self.preprocessing_transform(frames)
        if torch.all(processed_frames <= 1):
            processed_frames *= 255
        img_embedding = self.model(
            input=processed_frames.to(self.device), modality="vision"
        ).cpu()
        return img_embedding

    def compute_video_embed(self, frames):
        with torch.no_grad():
            video_embeds = self.batch_preprocess_frames(frames)
        if self.reward_type == "real_diff":
            return video_embeds[1:] - video_embeds[:-1]
        elif self.reward_type == "first_diff":
            return video_embeds[1:] - video_embeds[:1].repeat(
                (video_embeds.shape[0] - 1, 1)
            )
        return video_embeds


class CLIPVLMRewardFunction(VLMRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.device = config.device
        model_name = config.vlm
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _batch_preprocess_inputs(self, frames, lang):
        if torch.all(frames <= 1):
            frames *= 255
        inputs = self.processor(
            text=lang, images=frames, return_tensors="pt", padding=True
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def compute_video_embed(self, frames):
        with torch.no_grad():
            image_embeds = self.batch_preprocess_frames(frames)
        if self.reward_type == "real_diff":
            return image_embeds[1:] - image_embeds[:-1]
        elif self.reward_type == "first_diff":
            return image_embeds[1:] - image_embeds[:1].repeat(
                (image_embeds.shape[0] - 1, 1)
            )
        return image_embeds

    def _batch_preprocess_frames(self, frames):
        inputs = self._batch_preprocess_inputs(frames, ["dummy"])
        image_embeds = self.model(**inputs).image_embeds.cpu()
        return image_embeds

    def compute_per_timestep_lang_rewards(self, frames, lang, lang2=None):
        with torch.no_grad():
            total_lang = lang
            if lang2 is not None and "diff" in self.reward_type:
                assert len(lang) == len(lang2)
                total_lang = lang + lang2
            inputs = self._batch_preprocess_inputs(frames, total_lang)
            outputs = self.model(**inputs)
            logits_per_text = outputs.logits_per_text
            if self.reward_type == "real_diff":
                # do l2 of image difference
                text_embeds = outputs.text_embeds
                if lang2 is not None and "diff" in self.reward_type:
                    text_embeds = outputs.text_embeds[: len(lang)]
                    text_embeds2 = outputs.text_embeds[len(lang) :]
                    text_embeds = text_embeds2 - text_embeds
                image_embeds = outputs.image_embeds
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                image_difference = image_embeds - image_embeds[:, :1]
                image_difference = image_difference / image_difference.norm(
                    dim=-1, keepdim=True
                )
                # logits_per_text = text_embeds @ image_difference.T
                image_difference = image_difference.unsqueeze(0).repeat(
                    (text_embeds.shape[0], 1, 1)
                )
                text_embeds = text_embeds.unsqueeze(1).repeat(
                    (1, image_difference.shape[1], 1)
                )

                # create logits_per_text which will be the l2 distance between each image difference and each text embed
                logits_per_text = ((image_difference - text_embeds) ** 2).mean(dim=-1)
                assert logits_per_text.shape[0] == len(lang)
                assert logits_per_text.shape[0] == len(lang)
            probs = torch.softmax(logits_per_text, dim=0)
        return probs.cpu()


class XCLIPVLMRewardFunction(VLMRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.device = config.device
        model_name = config.vlm
        specific_model = model_name.split("/")[1]
        if "xclip-base-patch16-zero-shot" == specific_model:
            self.stack_size = 32
        elif (
            "xclip-base-patch16" == specific_model
            or "xclip-base-patch32" == specific_model
            or "xclip-large-patch14" == specific_model
        ):
            self.stack_size = 8
        elif (
            "xclip-base-patch16-16-frames" == specific_model
            or "xclip-base-patch32-16-frames" == specific_model
        ):
            self.stack_size = 16

        else:
            raise ValueError(f"Unknown model {model_name}")
        self.model = XCLIPModel.from_pretrained(model_name)
        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def compute_video_embed(self, frames):
        with torch.no_grad():
            all_embeds = []
            # turn the frames into a per-timestep video sequence
            # all_inputs = []
            for i in range(0, len(frames)):
                vid = frames[max(i - self.stack_size + 1, 0) : i + 1]
                # pad each of them with a copy of the first frame if needed
                if vid.shape[0] < self.stack_size:
                    vid = torch.cat(
                        [
                            vid[0]
                            .unsqueeze(0)
                            .repeat(self.stack_size - vid.shape[0], 1, 1, 1),
                            vid,
                        ],
                        dim=0,
                    )
                if torch.all(vid <= 1):
                    vid *= 255
                inputs = self.processor(
                    text=["dummy"], videos=list(vid), return_tensors="pt", padding=True
                )
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                # all_inputs.append(inputs)
                outputs = self.model(**inputs)
                embeds = outputs.video_embeds.cpu()
                all_embeds.append(embeds)
            # batched_inputs = {}
            # for key in inputs:
            #    batched_inputs[key] = torch.cat([i[key] for i in all_inputs], dim=0)
            # for key in batched_inputs:
            #    batched_inputs[key] = batched_inputs[key].to(self.device)
        embeds = torch.cat(all_embeds, dim=0)
        if self.reward_type == "real_diff":
            return embeds[1:] - embeds[:-1]
        elif self.reward_type == "first_diff":
            return embeds[1:] - embeds[:1].repeat((embeds.shape[0] - 1, 1))
        return embeds

    def compute_per_timestep_lang_rewards(self, frames, lang):
        with torch.no_grad():
            all_probs = []
            # turn the frames into a per-timestep video sequence
            for i in range(
                self.stack_size, len(frames) + self.stack_size, self.stack_size
            ):
                vid = frames[max(i - self.stack_size + 1, 0) : i + 1]
                original_len = len(vid)
                # pad each of them with a copy of the first frame if needed
                if vid.shape[0] < self.stack_size:
                    vid = torch.cat(
                        [
                            vid[0]
                            .unsqueeze(0)
                            .repeat(self.stack_size - vid.shape[0], 1, 1, 1),
                            vid,
                        ],
                        dim=0,
                    )
                if torch.all(vid <= 1):  # checks if not 0-255
                    vid *= 255
                inputs = self.processor(
                    text=lang, videos=list(vid), return_tensors="pt", padding=True
                )
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                outputs = self.model(**inputs)
                logits_per_text = outputs.logits_per_text
                assert logits_per_text.shape[0] == len(lang)
                probs = torch.softmax(logits_per_text, dim=0).cpu()
                all_probs.append(probs.repeat((1, original_len)))
        return torch.cat(all_probs, dim=-1)


class ViLTVLMRewardFunction(CLIPVLMRewardFunction, nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.device = config.device
        model_name = config.vlm
        self.model = ViltForImageAndTextRetrieval.from_pretrained(model_name)
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def compute_per_timestep_lang_rewards(self, frames, lang):
        logits_per_text = []
        # supports only 1 text at a time
        with torch.no_grad():
            for l in lang:
                l = [l] * frames.shape[0]
                if torch.all(frames <= 1):  # checks if not 0-255
                    frames *= 255
                inputs = self.processor(
                    text=l, images=frames, return_tensors="pt", padding=True
                )
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                outputs = self.model(**inputs)
                per_text_score = outputs.logits[:, 0].cpu()
                logits_per_text.append(per_text_score)
        return torch.softmax(torch.stack(logits_per_text), dim=0)

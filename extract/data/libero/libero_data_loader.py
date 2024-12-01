import os

from extract.utils.general_utils import AttrDict
from extract.data.kitchen.src.kitchen_data_loader import (
    SkillClusterD4RLSequenceSplitDataset,
    D4RLSequenceSplitDataset,
    UVDD4RLDataset,
)


def get_updated_data_dir(data_dir, phase, finetune_dataset=None):
    # expect data_dir to just be a folder pointing to the path to be filled in later
    if finetune_dataset:
        return os.path.join(data_dir, f"libero_{finetune_dataset}")
    if phase == "train":
        # updated_data_dir = os.path.join(data_dir, "libero_90.hdf5")
        updated_data_dir = os.path.join(data_dir, "libero_90")  # lmdb
    elif phase == "val":
        # updated_data_dir = os.path.join(data_dir, "libero_10.hdf5")
        updated_data_dir = os.path.join(data_dir, "libero_10")  # lmdb
    return updated_data_dir


def get_updated_cluster_dir(cluster_file_dir, phase, finetune_dataset=None):
    # expect cluster_file_dir to just be a folder pointing to the path to be filled in later
    cluster_file_dir = cluster_file_dir.replace("90/clusters.h5", "")
    # in because the dataspec is literally overwritten and we use it again for the other dataloader for train/val
    cluster_file_dir = cluster_file_dir.replace("10/clusters.h5", "")
    # in because the dataspec is literally overwritten and we use it again for the other dataloader for train/val
    cluster_file_dir = cluster_file_dir.replace(f"{finetune_dataset}/clusters.h5", "")
    if finetune_dataset:
        return os.path.join(cluster_file_dir, finetune_dataset, "clusters.h5")
    if phase == "train":
        updated_cluster_file_dir = os.path.join(cluster_file_dir, "90", "clusters.h5")
    elif phase == "val":
        updated_cluster_file_dir = os.path.join(cluster_file_dir, "10", "clusters.h5")
    return updated_cluster_file_dir


class LIBEROSequenceSplitDataset(D4RLSequenceSplitDataset):
    def __init__(
        self,
        data_dir,
        data_conf,
        phase,
        resolution=None,
        shuffle=True,
        dataset_size=-1,
    ):

        if phase == "train":
            self.SPLIT = AttrDict(train=1.00, val=0.0, test=0.0)
        else:
            self.SPLIT = AttrDict(train=0.00, val=1.0, test=0.0)
        data_conf.dataset_spec.load_data = True
        # assert data_conf.dataset_spec.use_image is True
        # assert data_conf.dataset_spec.use_image is True
        dataset = None
        if "finetune" in data_conf and data_conf.finetune == True:
            dataset = data_conf.finetune_dataset
        updated_data_dir = get_updated_data_dir(data_dir, phase, dataset)
        return super().__init__(
            updated_data_dir,
            data_conf,
            phase,
            resolution=resolution,
            shuffle=shuffle,
            dataset_size=dataset_size,
        )


class SkillClusterLIBEROSequenceSplitDataset(SkillClusterD4RLSequenceSplitDataset):
    SPLIT = AttrDict(train=1.00, val=0.00, test=0.0)

    def __init__(
        self,
        data_dir,
        data_conf,
        phase,
        resolution=None,
        shuffle=True,
        dataset_size=-1,
    ):
        if phase == "train":
            self.SPLIT = AttrDict(train=1.00, val=0.0, test=0.0)
        else:
            self.SPLIT = AttrDict(train=0.00, val=1.0, test=0.0)
        # assert data_conf.dataset_spec.use_image is True
        data_conf.dataset_spec.load_data = True
        dataset = None
        if "finetune" in data_conf and data_conf.finetune == True:
            dataset = data_conf.finetune_dataset
        updated_data_dir = get_updated_data_dir(data_dir, phase, dataset)
        updated_cluster_file_dir = get_updated_cluster_dir(
            data_conf.dataset_spec.cluster_data_file, phase, dataset
        )
        data_conf.dataset_spec.cluster_data_file = updated_cluster_file_dir
        return super().__init__(
            updated_data_dir,
            data_conf,
            phase,
            resolution=resolution,
            shuffle=shuffle,
            dataset_size=dataset_size,
        )


class UVDSkillClusterLIBEROSequenceSplitDataset(UVDD4RLDataset):
    SPLIT = AttrDict(train=1.00, val=0.00, test=0.0)

    def __init__(
        self,
        data_dir,
        data_conf,
        phase,
        resolution=None,
        shuffle=True,
        dataset_size=-1,
    ):
        if phase == "train":
            self.SPLIT = AttrDict(train=1.00, val=0.0, test=0.0)
        else:
            self.SPLIT = AttrDict(train=0.00, val=1.0, test=0.0)
        # assert data_conf.dataset_spec.use_image is True
        data_conf.dataset_spec.load_data = True
        dataset = None
        if "finetune" in data_conf and data_conf.finetune == True:
            dataset = data_conf.finetune_dataset
        updated_data_dir = get_updated_data_dir(data_dir, phase, dataset)
        updated_cluster_file_dir = get_updated_cluster_dir(
            data_conf.dataset_spec.cluster_data_file, phase, dataset
        )
        data_conf.dataset_spec.cluster_data_file = updated_cluster_file_dir
        return super().__init__(
            updated_data_dir,
            data_conf,
            phase,
            resolution=resolution,
            shuffle=shuffle,
            dataset_size=dataset_size,
        )

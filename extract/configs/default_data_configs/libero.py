from extract.utils.general_utils import AttrDict
from extract.data.libero.libero_data_loader import LIBEROSequenceSplitDataset


data_spec = AttrDict(
    dataset_class=LIBEROSequenceSplitDataset,
    n_actions=7,
    state_dim=1,
    env_name=None,
    res=84,
    # crop_rand_subseq=True,
    use_image=False,  # state based libero for now
    image_aug=True,
    no_load_in_ram=True,
    image_aug_specs=AttrDict(
        # Taken from LIBERO/code for TAIL: Task specfic adapters for imitation learning
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        color_p=0.9,
        noise_std=0.1,
        noise_p=0.0,
        channel_shuffle_p=0.0,
        degrees=15,
        translate=0.1,
        affine_p=0.6,
        erase_p=0.1,
    ),
)
# data_spec.max_seq_len = 280

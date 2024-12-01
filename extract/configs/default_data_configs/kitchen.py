from extract.utils.general_utils import AttrDict
from extract.data.kitchen.src.kitchen_data_loader import D4RLSequenceSplitDataset


data_spec = AttrDict(
    dataset_class=D4RLSequenceSplitDataset,
    n_actions=9,
    state_dim=60,
    env_name="kitchen-mixed-v0",
    res=64,
    crop_rand_subseq=True,
    use_image=False,
)
data_spec.max_seq_len = 280

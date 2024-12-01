import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the config file directory")
    parser.add_argument(
        "--run_name",
        default=None,
        required=True,
        type=str,
        help="name of the run for wandB logging. If none, will be set to a default name.",
    )
    parser.add_argument(
        "--run_group",
        default=None,
        type=str,
        help="name of the run group for wandB logging for easier grouping/metrics tracking (optional)",
    )
    # Folder settings
    parser.add_argument(
        "--prefix",
        help="experiment prefix, if given creates subfolder in experiment directory. If not specified and you specify a run_name, it'll default to the run_name",
    )
    parser.add_argument(
        "--new_dir",
        default=False,
        type=int,
        help="If True, concat datetime string to exp_dir.",
    )
    parser.add_argument(
        "--dont_save",
        default=False,
        type=int,
        help="if True, nothing is saved to disk. Note: this doesn't work",
    )  # TODO this doesn't work

    # Running protocol
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--train",
        default=True,
        type=int,
        help="if False, will run one validation epoch",
    )
    parser.add_argument(
        "--test_prediction",
        default=True,
        type=int,
        help="if False, prediction isn't run at validation time",
    )
    parser.add_argument(
        "--skip_first_val",
        default=False,
        type=int,
        help="if True, will skip the first validation epoch",
    )
    parser.add_argument(
        "--val_sweep",
        default=False,
        type=int,
        help="if True, runs validation on all existing model checkpoints",
    )

    # Misc
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="will set CUDA_VISIBLE_DEVICES to selected value",
    )
    parser.add_argument(
        "--strict_weight_loading",
        default=True,
        type=int,
        help="if True, uses strict weight loading function",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="overrides config/default seed for more convenient seed setting.",
    )
    parser.add_argument(
        "--log_interval",
        default=500,
        type=int,
        help="number of updates per training log",
    )
    parser.add_argument(
        "--per_epoch_img_logs",
        default=1,
        type=int,
        help="number of image loggings per epoch",
    )
    parser.add_argument(
        "--val_data_size",
        default=-1,
        type=int,
        help="number of sequences in the validation set. If -1, the full dataset is used",
    )
    parser.add_argument(
        "--val_interval", default=5, type=int, help="number of epochs per validation"
    )
    parser.add_argument(
        "--config_override",
        default="",
        type=str,
        help='override to config file in format "key1.key2=val1,key3=val2"',
    )

    # Debug
    parser.add_argument(
        "--detect_anomaly",
        default=False,
        type=int,
        help="if True, uses autograd.detect_anomaly()",
    )
    parser.add_argument(
        "--feed_random_data",
        default=False,
        type=int,
        help="if True, we feed random data to the model to test its performance",
    )
    parser.add_argument(
        "--train_loop_pdb",
        default=False,
        type=int,
        help="if True, opens a pdb into training loop",
    )
    parser.add_argument(
        "--debug", default=False, type=int, help="if True, runs in debug mode"
    )

    # add kl_div_weight
    parser.add_argument(
        "--save2mp4",
        default=False,
        type=bool,
        help="if set, videos will be saved locally",
    )

    parsed = parser.parse_args()
    if parsed.seed is None:
        parsed.seed = np.random.randint(100000)
    # if parsed.prefix is None and parsed.run_group is not None:
    #    parsed.prefix = parsed.run_group + "_" + str(parsed.seed)
    return parsed

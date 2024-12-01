import os
from contextlib import contextmanager
import numpy as np
from gym import wrappers
import random
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

from extract.utils.general_utils import ParamDict
from extract.rl.components.environment import GymEnv

from sentence_transformers import SentenceTransformer

IMAGE_KEY = "agentview_image"

VIDEO_RENDER_RES = 300
NUM_TASKS = 10  # MAX is 10
EVALS_PER_TASK = 3
# RENDER_MAP = {0: 3, 1: 2, 2: 1, 3: 0, 4: 7, 5: 6, 6: 5, 7: 4}  # KAIST A
# RENDER_MAP = {0 : 2, 1 : 3, 2 : 1, 3 : 0, 4 : 6, 5 : 7, 6: 5, 7 : 4} # kaist b
RENDER_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}  # DEFAULT


class LIBEROEnv(GymEnv):
    def __init__(self, config):
        config.name = self.task_id = config.task_id
        self.task_suite_name = (
            config.task_suite
        )  # can also choose libero_spatial, libero_object, libero_goal, libero_10, libero_90
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.task_suite_name]()
        self._randomly_sample_task = False
        if self.task_id is None:
            self._randomly_sample_task = True
            # reset the seed to the current upon init
            random.seed()
            config.name = random.randint(
                0, self.num_tasks - 1
            )  # name is passed to make-en
            self.task_id = config.name
        else:
            print(f"Hardcoding setting task to {self.task_id}")

        self._completed_goal_states = set()

        config.name = self.task_id = int(self.task_id)  # float from override
        # self.task_suite_name = "libero_10"  # can also choose libero_spatial, libero_object, etc. Libero_10 is what we're evaluating on though
        self._obs = None
        self._render_obs = None
        self._dummy_obs = np.array([1])
        self._lang_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.lang = None
        self._val_mode = False
        self._just_switched_to_val_mode = False
        super().__init__(config)

    def step(self, action):
        """Takes an action in the environment."""
        obs, reward, done, info = super().step(action)
        info["task_description"] = self._task_description
        reward, success = self._get_any_reward(obs, reward, done, info)
        info["success"] = success
        # info["lang"] = self.lang
        return obs, np.float64(reward), done, info

    def _get_any_reward(self, obs, reward, done, info):
        """Returns reward from the environment for ANY goal completed."""
        goal_state = self._env.env.parsed_problem["goal_state"]
        total_reward = 0
        success = True
        # success should be true whenever all goal states are satisfied, not just one and not relaiant on self._completed_goal_states
        for state in goal_state:
            if self._env.env._eval_predicate(state):  # checks goal
                this_success = True
                if (
                    tuple(state) not in self._completed_goal_states
                ):  # checks not already completed
                    total_reward += 1
                    self._completed_goal_states.add(tuple(state))
            else:
                this_success = False
            success = success and this_success
        return total_reward, success

    def _wrap_observation(self, obs):
        """Process raw observation from the environment before return."""
        self._obs = np.asarray(obs[IMAGE_KEY], dtype=np.float32)
        # flip the observation because it comes reversed from the environment... lol
        self._obs = self._obs[::-1] / 255.0
        self._render_obs = self._obs
        if self._val_mode:
            # downsize the original obs to the screen_height and screen_width
            self._obs = (
                np.asarray(
                    Image.fromarray((self._obs * 255).astype(np.uint8)).resize(
                        (self._hp.screen_height, self._hp.screen_width)
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )

        # return dummy state to work with spirl existing code
        return self._dummy_obs

    def _default_hparams(self):
        return (
            super()
            ._default_hparams()
            .overwrite(
                ParamDict(
                    {
                        "randomize_initial_state": True,
                        "task_id": None,
                    }
                )
            )
        )

    def reset(self):
        """Resets the environment and returns the initial observation."""
        # we're sampling a training environment, then we want to randomly sample a new task ID
        # however, if self._val_mode is true then we want to keep the task id fixed because it's given beforehand for consistent evaluation
        if not self._val_mode and self._randomly_sample_task:
            self.task_id = random.randint(0, self.num_tasks - 1)
        if (
            self._env is None
            or self._randomly_sample_task
            or self._val_mode
            or self._just_switched_to_val_mode
        ):
            if self._just_switched_to_val_mode:
                self._just_switched_to_val_mode = False
            # if env is none its the first time
            # if ranodmly sample task we need to re-init the env every time
            # if self._val_mode we need to re-init the env to render at a higher resolution for saving rollouts
            # if just switched to val mode, we need to re-make the env to render at a lower resolution again.
            self._env = self._make_env(self.task_id)
        self._completed_goal_states = set()
        init_states = self.task_suite.get_task_init_states(
            self.task_id
        )  # for benchmarking purpose, we fix the set of initial states
        if self._hp.randomize_initial_state:
            self._env.reset()
            init_state_id = random.randint(0, len(init_states) - 1)
            obs = self._env.set_init_state(init_states[init_state_id])
            return self._wrap_observation(obs)
        else:
            init_state_id = 0
            self._env.set_init_state(init_states[init_state_id])
            return super().reset()

    def _make_env(self, task_id: int):
        """Instantiates the environment given the ID of the task."""
        # retrieve a specific task
        # assert task_id >= 0 and task_id < 10
        task = self.task_suite.get_task(task_id)
        # task_name = task.name
        task_description = task.language
        self._task_description = task_description
        self.lang = np.expand_dims(
            self._lang_embed_model.encode([task_description]), 0
        )  # [1, 1, lang_dim] for time dimension keeping for compat
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        # print(
        #    f"[info] retrieving task {task_id} from suite {self.task_suite_name}, the "
        #    + f"language instruction is {task_description}, and the bddl file is {task_bddl_file}"
        # )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": (
                self._hp.screen_height if not self._val_mode else VIDEO_RENDER_RES
            ),
            "camera_widths": (
                self._hp.screen_width if not self._val_mode else VIDEO_RENDER_RES
            ),
            "camera_names": "agentview",
            "render_gpu_device_id": RENDER_MAP[self._hp.gpu],
        }

        env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        if isinstance(env, wrappers.TimeLimit) and self._hp.unwrap_time:
            # unwraps env to avoid this bug: https://github.com/openai/gym/issues/1230
            env = env.env
        return env

    def render(self, mode=None):
        if mode == "rgb_array":
            return self._obs
        return self._render_obs

    @contextmanager
    def val_mode(self, env_id=None):
        """Sets validation parameters if desired. To be used like: with env.val_mode(): ...<do something>..."""
        self._val_mode = True
        if self._randomly_sample_task:
            if env_id is not None:
                self.task_id = (
                    env_id % self.num_tasks
                )  # make sure the task id is within the range
                print(f"EVAL: setting task_id to {self.task_id}")
            else:
                self.task_id = None
        yield
        self._val_mode = False
        self._just_switched_to_val_mode = True
        if self._randomly_sample_task:
            self.task_id = random.randint(0, self.num_tasks - 1)

    @property
    def num_tasks(self):
        if not self._randomly_sample_task:
            return 1
        return NUM_TASKS

    @property
    def num_eval_tasks(self):
        """Returns number of evaluation tasks. Defaults to None if not set. Used for eval loop."""
        if self._hp.randomize_initial_state:
            return self.num_tasks * EVALS_PER_TASK
        return self.num_tasks
        # return len(self.task_suite)

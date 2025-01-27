# Copyright 2021-2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train a TG / TXL (trees) / TXL (words) model."""


# This needs to be put first -- it prevents TF from allocating GPU memory.
import os

os.environ["TF_ENABLED_DEVICE_TYPES"] = "CPU"

# pylint: disable=g-import-not-at-top,g-bad-import-order
import functools
from absl import app
from absl import flags
from ml_collections import config_flags
from transformer_grammars.training import train


_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", None, "Override for training batch size.")
flags.DEFINE_integer("num_layers", None, "Override for the number of model layers.")
flags.DEFINE_float("max_lr", None, "Override for max learning rate in the schedule.")
flags.DEFINE_string(
    "output_config", "configs/modified_config.json", "Path to save the modified config."
)


def override_config(config, output_path):
    if _FLAGS.batch_size is not None:
        config.training.batch_size = _FLAGS.batch_size
    if _FLAGS.num_layers is not None:
        config.model.num_layers = _FLAGS.num_layers
    if _FLAGS.max_lr is not None:
        config.training.lr_schedule.kwargs.max_lr = _FLAGS.max_lr

    # Save the modified config to a file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config.to_json())
    print(f"Modified config saved to {output_path}")
    return config


def main(argv):
    config = _CONFIG.value
    config = override_config(config, _FLAGS.output_config)
    functools.partial(train.main, config)


if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(main)

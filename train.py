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

flags.DEFINE_string(
    "sentencepiece_vocab_filename", None, "Override for sentencepiece_vocab_filename."
)
flags.DEFINE_string(
    "training_dataset_filename", None, "Override for training dataset filename."
)
flags.DEFINE_string(
    "evaluation_dataset_filename", None, "Override for evaluation dataset filename."
)
flags.DEFINE_string(
    "lr_schedule_name", None, "Override for learning rate scheduler name."
)
flags.DEFINE_float("start_lr", None, "Override for starting learning rate.")
flags.DEFINE_integer("training_batch_size", None, "Override for batch size.")
flags.DEFINE_integer("warmup_steps", None, "Override for warmup steps.")
flags.DEFINE_integer("cosine_cycle_length", None, "Override for cosine cycle length.")
flags.DEFINE_integer("vocab_size", None, "Override for vocabulary size.")
flags.DEFINE_integer("num_layers", None, "Override for the number of layers.")
flags.DEFINE_integer("ffw_hidden_size", None, "Override for feed-forward hidden size.")
flags.DEFINE_integer("d_model", None, "Override for the model dimension.")
flags.DEFINE_string("checkpoint_path", None, "Override for checkpointing path.")
flags.DEFINE_string("logging_entity", None, "Override for logging entity.")
flags.DEFINE_string("logging_project", None, "Override for logging project.")
flags.DEFINE_string("logging_group", None, "Override for logging group.")
flags.DEFINE_string("logging_run_name", None, "Override for logging run name.")
flags.DEFINE_string(
    "output_config", "configs/modified_config.json", "Path to save the modified config."
)
flags.DEFINE_list("logging_tags", None, "Override for logging tags.")


def override_config(config, output_path):
    # Training dataset overrides
    if _FLAGS.sentencepiece_vocab_filename:
        config.sentencepiece_vocab_filename = _FLAGS.sentencepiece_vocab_filename
    if _FLAGS.training_batch_size:
        config.training.batch_size = _FLAGS.training_batch_size
    if _FLAGS.training_dataset_filename:
        config.training.dataset.kwargs.filename = _FLAGS.training_dataset_filename
    if _FLAGS.evaluation_dataset_filename:
        config.evaluation.dataset.kwargs.filename = _FLAGS.evaluation_dataset_filename

    # Learning rate scheduler overrides
    if _FLAGS.lr_schedule_name:
        config.training.lr_schedule.name = _FLAGS.lr_schedule_name
    if _FLAGS.warmup_steps is not None:
        config.training.lr_schedule.kwargs.warmup_steps = _FLAGS.warmup_steps
    if _FLAGS.start_lr is not None:
        config.training.lr_schedule.kwargs.start_lr = _FLAGS.start_lr
    if _FLAGS.cosine_cycle_length is not None:
        config.training.lr_schedule.kwargs.cosine_cycle_length = (
            _FLAGS.cosine_cycle_length
        )

    # Model overrides
    if _FLAGS.vocab_size is not None:
        config.model.vocab_size = _FLAGS.vocab_size
    if _FLAGS.num_layers is not None:
        config.model.num_layers = _FLAGS.num_layers
    if _FLAGS.ffw_hidden_size is not None:
        config.model.ffw_hidden_size = _FLAGS.ffw_hidden_size
    if _FLAGS.d_model is not None:
        config.model.d_model = _FLAGS.d_model

    # Checkpointing overrides
    if _FLAGS.checkpoint_path:
        config.checkpointing.path = _FLAGS.checkpoint_path

    # Logging overrides
    if _FLAGS.logging_entity:
        config.logging.entity = _FLAGS.logging_entity
    if _FLAGS.logging_project:
        config.logging.project = _FLAGS.logging_project
    if _FLAGS.logging_group:
        config.logging.group = _FLAGS.logging_group
    if _FLAGS.logging_run_name:
        config.logging.run_name = _FLAGS.logging_run_name
    if _FLAGS.logging_tags:
        config.logging.tags = _FLAGS.logging_tags

    # Save modified config
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config.to_json())
    print(f"Modified config saved to {output_path}")
    return config


def main(argv):
    config = _CONFIG.value
    config = override_config(config, _FLAGS.output_config)
    train.main(config, None)  # defined with second argument not being used


if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(main)

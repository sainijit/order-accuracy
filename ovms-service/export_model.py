#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
import jinja2
import json
import shutil
import tempfile
from pathlib import Path

def add_common_arguments(parser):
    parser.add_argument('--model_repository_path', required=False, default='models_vlm', help='Where the model should be exported to', dest='model_repository_path')
    parser.add_argument('--source_model', required=True, help='HF model name or path to the local folder with PyTorch or OpenVINO model', dest='source_model')
    parser.add_argument('--model_name', required=False, default=None, help='Model name that should be used in the deployment. Equal to source_model if HF model name is used', dest='model_name')
    parser.add_argument('--weight-format', default='int4', help='precision of the exported model', dest='precision')
    parser.add_argument('--config_file_path', default='config.json', help='path to the config file', dest='config_file_path')
    parser.add_argument('--overwrite_models', default=False, action='store_true', help='Overwrite the model if it already exists in the models repository', dest='overwrite_models')
    parser.add_argument('--target_device', default="GPU", help='CPU, GPU, NPU or HETERO, default is GPU', dest='target_device')
    parser.add_argument('--ov_cache_dir', default=None, help='Folder path for compilation cache to speedup initialization time', dest='ov_cache_dir')
    parser.add_argument('--extra_quantization_params', required=False, help='Add advanced quantization parameters. Check optimum-intel documentation. Example: "--sym --group-size -1 --ratio 1.0 --awq --scale-estimation --dataset wikitext2"', dest='extra_quantization_params')

parser = argparse.ArgumentParser(description='Export Hugging face models to OVMS models repository including all configuration for deployments')

subparsers = parser.add_subparsers(help='subcommand help', required=True, dest='task')
parser_text = subparsers.add_parser('text_generation', help='export model for chat and completion endpoints')
add_common_arguments(parser_text)
parser_text.add_argument('--pipeline_type', default=None, choices=["LM", "LM_CB", "VLM", "VLM_CB", "AUTO"], help='Type of the pipeline to be used. AUTO is used by default', dest='pipeline_type')
parser_text.add_argument('--kv_cache_precision', default=None, choices=["u8"], help='u8 or empty (model default). Reduced kv cache precision to u8 lowers the cache size consumption.', dest='kv_cache_precision')
parser_text.add_argument('--enable_prefix_caching', action='store_true', help='This algorithm is used to cache the prompt tokens.', dest='enable_prefix_caching')
parser_text.add_argument('--disable_dynamic_split_fuse', action='store_false', help='The maximum number of tokens that can be batched together.', dest='dynamic_split_fuse')
parser_text.add_argument('--max_num_batched_tokens', default=None, help='empty or integer. The maximum number of tokens that can be batched together.', dest='max_num_batched_tokens')
parser_text.add_argument('--max_num_seqs', default=None, help='256 by default. The maximum number of sequences that can be processed together.', dest='max_num_seqs')
parser_text.add_argument('--cache_size', default=10, type=int, help='KV cache size in GB', dest='cache_size')
parser_text.add_argument('--draft_source_model', required=False, default=None, help='HF model name or path to the local folder with PyTorch or OpenVINO draft model. '
                         'Using this option will create configuration for speculative decoding', dest='draft_source_model')
parser_text.add_argument('--draft_model_name', required=False, default=None, help='Draft model name that should be used in the deployment. '
                         'Equal to draft_source_model if HF model name is used', dest='draft_model_name')

args = parser.parse_args()

# For order-accuracy, we're using Qwen2-VL-2B-Instruct model
# This script is provided for reference. To export the model, run:
# python export_model.py text_generation --source_model Qwen/Qwen2-VL-2B-Instruct --weight-format int4 --target_device GPU

print(f"Export script for OVMS models")
print(f"To export Qwen2-VL-2B-Instruct model, run:")
print(f"python export_model.py text_generation --source_model Qwen/Qwen2-VL-2B-Instruct --weight-format int4 --target_device GPU")
print(f"\nNote: This requires the full implementation from optimum-intel package.")
print(f"The model will be exported to: {args.model_repository_path}")

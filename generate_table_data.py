#!/usr/bin/env python3
"""
Script to generate accuracy table data from evaluation results.
Generates mean ± std over 3 seeds for Llama-3.1-8B IT across different top-k% values
for GSM8K (English, German, Hindi) and RACE (English, German, Hindi) datasets.
"""

import json
import os
import re
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# Dataset configurations
# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race_german/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/scale/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race_german/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/scale/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }



# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/scale/results_gsm8k_race_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_race_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     },
#     'code': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_codealpaca_race/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': "oracle_acc",
#         'task_key': "race",
#         'label': 'Code',
#         'dataset_type': 'code',
#         'primary_task_key': 'codealpaca',
#         'primary_metric_candidates': ['oracle_acc'],
#         'primary_flex_metric_candidates': [],
#         'oracle_metric_file_prefix': 'codealpaca_oracle_metrics',
#         'include_flex': False
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_german/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/prune/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_german/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/prune/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }


# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/prune/results_gsm8k_race_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     },
#     'code': {
#         'path': "/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_codealpaca_race/eval_results/Qwen/Qwen3-4B-Instruct-2507",
#         'gsm8k_key': "oracle_acc",
#         'task_key': "race",
#         'label': 'Code',
#         'dataset_type': 'code',
#         'primary_task_key': 'codealpaca',
#         'primary_metric_candidates': ['oracle_acc'],
#         'primary_flex_metric_candidates': [],
#         'oracle_metric_file_prefix': 'codealpaca_oracle_metrics',
#         'include_flex': False
#     }
# }

###################### MMLU #########################
# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_german/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_german/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_hindi_max300/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/scale/results_gsm8k_mmlu_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_gsm8k_mmlu_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     },
#     'code': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/scale/results_codealpaca_mmlu/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': "oracle_acc",
#         'task_key': "mmlu",
#         'label': 'Code',
#         'dataset_type': 'code',
#         'primary_task_key': 'codealpaca',
#         'primary_metric_candidates': ['oracle_acc'],
#         'primary_flex_metric_candidates': [],
#         'oracle_metric_file_prefix': 'codealpaca_oracle_metrics',
#         'include_flex': False
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_german/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_german/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_hindi_max300/eval_results/meta-llama/Llama-3.2-1B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_mmlu_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     },
#     'code': {
#         'path': "/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_codealpaca_mmlu/eval_results/Qwen/Qwen3-4B-Instruct-2507",
#         'gsm8k_key': "oracle_acc",
#         'task_key': "mmlu",
#         'label': 'Code',
#         'dataset_type': 'code',
#         'primary_task_key': 'codealpaca',
#         'primary_metric_candidates': ['oracle_acc'],
#         'primary_flex_metric_candidates': [],
#         'oracle_metric_file_prefix': 'codealpaca_oracle_metrics',
#         'include_flex': False
#     }
# }

####################### Pretrain #########################
# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_race/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_race_german/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/prune/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_race/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_race_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_race_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/prune/results_gsm8k_race_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }


    ### MMLU ####
# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_mmlu/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_mmlu_german/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_mmlu_hindi_max300/eval_results/meta-llama/Llama-3.1-8B-Instruct',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     }
# }

# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_mmlu/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'mmlu',
#         'label': 'English'
#     },
#     'german': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_mmlu_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'mmlu_de',
#         'label': 'German'
#     },
#     'hindi': {
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/scale/results_gsm8k_mmlu_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/pretrain/results_gsm8k_mmlu_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'mmlu_hi_max300',
#         'label': 'Hindi'
#     }
# }

##### weird effect ####
# DATASETS = {
#     'english': {
#         'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/weird_effect/results_gsm8k_race/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_cot',
#         'task_key': 'race',
#         'label': 'English'
#     },
#     'german': {
#         'path': 'fds/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_german/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_de_cot',
#         'task_key': 'race_de',
#         'label': 'German'
#     },
#     'hindi': {
#         # 'path': '/raid/s3/opengptx/behzad_shomali/LabTest/MathNeuro/results/prune/results_gsm8k_race_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'path': '/rafsdid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_hindi_max300/eval_results/Qwen/Qwen3-4B-Instruct-2507',
#         'gsm8k_key': 'gsm8k_hi_cot_max300',
#         'task_key': 'race_hi_max300',
#         'label': 'Hindi'
#     }
# }


###################### Individual vs group effect #########################

DATASETS = {
    'english': {
        'path': "/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_random/eval_results/meta-llama/Llama-3.2-1B-Instruct",
        'gsm8k_key': 'gsm8k_cot',
        'task_key': 'race',
        'label': 'English'
    },
    'german': {
        'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_german_random/eval_results/meta-llama/Llama-3.2-1B-Instruct',
        'gsm8k_key': 'gsm8k_de_cot',
        'task_key': 'race_de',
        'label': 'German'
    },
    'hindi': {
        'path': '/raid/s3/opengptx/behzad_shomali/LabTest/results/prune/results_gsm8k_race_hindi_max300_random/eval_results/meta-llama/Llama-3.2-1B-Instruct',
        'gsm8k_key': 'gsm8k_hi_cot_max300',
        'task_key': 'race_hi_max300',
        'label': 'Hindi'
    }
}


# Top-k percentages to evaluate
TOP_K_VALUES = [0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15]
NUM_RUNS = 3 


def load_json(filepath: str) -> Dict:
    """Load and parse a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def _to_float(value: Any) -> Optional[float]:
    """Convert value to float if possible."""
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metric_value(container: Dict[str, Any], metric_candidates: List[str]) -> Optional[float]:
    """Extract metric value from dict using candidate keys (with case-insensitive fallback)."""
    if not isinstance(container, dict):
        return None

    for key in metric_candidates:
        if key in container:
            value = _to_float(container[key])
            if value is not None:
                return value

    lower_map = {k.lower(): v for k, v in container.items() if isinstance(k, str)}
    for key in metric_candidates:
        value = _to_float(lower_map.get(key.lower()))
        if value is not None:
            return value

    return None


def _task_key_matches(expected: str, candidate: str) -> bool:
    """Fuzzy match for task keys across format variants."""
    expected = expected.lower()
    candidate = candidate.lower()

    if expected == candidate:
        return True

    if expected.startswith("race") and candidate.startswith("race"):
        return True
    if expected.startswith("mmlu") and candidate.startswith("mmlu"):
        return True
    if expected.startswith("gsm8k") and candidate.startswith("gsm8k"):
        return True

    return False


def _extract_from_json(data: Dict[str, Any], task_key: str, metric_candidates: List[str]) -> Optional[float]:
    """Extract metric from JSON with robust task-key and structure fallback."""
    if not isinstance(data, dict):
        return None

    # 1) Direct task key
    if task_key in data and isinstance(data[task_key], dict):
        value = _extract_metric_value(data[task_key], metric_candidates)
        if value is not None:
            return value

    # 2) Fuzzy task key
    for key, value_dict in data.items():
        if isinstance(key, str) and isinstance(value_dict, dict) and _task_key_matches(task_key, key):
            value = _extract_metric_value(value_dict, metric_candidates)
            if value is not None:
                return value

    # 3) Single nested dict fallback
    nested_dict_values = [v for v in data.values() if isinstance(v, dict)]
    if len(nested_dict_values) == 1:
        value = _extract_metric_value(nested_dict_values[0], metric_candidates)
        if value is not None:
            return value

    # 4) Root-level metric fallback
    return _extract_metric_value(data, metric_candidates)


def _find_matching_file(base_path: str, run: int, topk: float, suffix: str, prefixes: List[str]) -> Optional[str]:
    """Find matching top-k file by parsed calculate value."""
    for prefix in prefixes:
        # Try _run{run} pattern first
        pattern = f"{prefix}_calculate*_run{run}{suffix}.json"
        candidates = sorted(Path(base_path).glob(pattern))
        for candidate in candidates:
            match = re.search(r"calculate([0-9.eE-]+)_scalar", candidate.name)
            if not match:
                continue
            try:
                candidate_topk = float(match.group(1))
            except ValueError:
                continue
            if np.isclose(candidate_topk, topk, rtol=0.0, atol=1e-12):
                return str(candidate)

    # Fallback: try _random_rseed* pattern (sorted, pick run-th match)
    for prefix in prefixes:
        pattern = f"{prefix}_calculate*_random_rseed*.json"
        candidates = sorted(Path(base_path).glob(pattern))
        matched = []
        for candidate in candidates:
            # Filter by suffix: when suffix is "" exclude _train_task files,
            # when suffix is "_train_task" only include those files
            if suffix:
                if not candidate.stem.endswith(suffix.rstrip('.json')):
                    # Check if the filename (without .json) ends with the suffix
                    name_no_ext = candidate.name.rsplit('.json', 1)[0]
                    if not name_no_ext.endswith(suffix):
                        continue
            else:
                # suffix="" means we want the base file, not _train_task
                name_no_ext = candidate.name.rsplit('.json', 1)[0]
                if name_no_ext.endswith('_train_task'):
                    continue

            match = re.search(r"calculate([0-9.eE-]+)_scalar", candidate.name)
            if not match:
                continue
            try:
                candidate_topk = float(match.group(1))
            except ValueError:
                continue
            if np.isclose(candidate_topk, topk, rtol=0.0, atol=1e-12):
                matched.append(str(candidate))
        if run < len(matched):
            return matched[run]

    return None


def get_dataset_columns(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return output columns for a dataset block."""
    label = config['label']

    if config.get('dataset_type') == 'code':
        second_task_name = "MMLU" if "mmlu" in config.get('task_key', '').lower() else "RACE"
        return [
            {'score_key': 'primary', 'header': 'Code', 'latex_subheader': 'Code'},
            {'score_key': 'secondary', 'header': f'{second_task_name}_{label}', 'latex_subheader': second_task_name},
        ]

    second_task_name = "MMLU" if "mmlu" in config.get('task_key', '').lower() else "RACE"
    cols = [
        {'score_key': 'primary', 'header': f'GSM8K_{label}', 'latex_subheader': 'GSM8K'},
    ]
    if config.get('include_flex', True):
        cols.append({'score_key': 'primary_flex', 'header': f'GSM8K_{label}_flex', 'latex_subheader': 'GSM8K flex'})
    cols.append({'score_key': 'secondary', 'header': f'{second_task_name}_{label}', 'latex_subheader': second_task_name})
    return cols


def get_pretrain_scores(config: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract pre-training scores from pre_results files with format fallbacks."""
    base_path = config['path']
    primary_task_key = config.get('primary_task_key', config['gsm8k_key'])
    task_key = config['task_key']
    primary_metric_candidates = config.get('primary_metric_candidates', ['exact_match,strict-match'])
    primary_flex_metric_candidates = config.get('primary_flex_metric_candidates', ['exact_match,flexible-extract'])
    secondary_metric_candidates = config.get('secondary_metric_candidates', ['acc,none'])

    scores: Dict[str, Optional[float]] = {
        'primary': None,
        'primary_flex': None,
        'secondary': None,
    }

    # Primary task metrics (usually from pre_results_train_task.json)
    train_file = os.path.join(base_path, 'pre_results_train_task.json')
    if os.path.exists(train_file):
        try:
            data = load_json(train_file)
            scores['primary'] = _extract_from_json(data, primary_task_key, primary_metric_candidates)
            if config.get('include_flex', True):
                scores['primary_flex'] = _extract_from_json(data, primary_task_key, primary_flex_metric_candidates)
        except (json.JSONDecodeError, ValueError):
            pass

    # Optional oracle metrics fallback (e.g., codealpaca_oracle_metrics.json)
    if scores['primary'] is None and config.get('oracle_metric_file_prefix'):
        oracle_file = os.path.join(base_path, f"{config['oracle_metric_file_prefix']}.json")
        if os.path.exists(oracle_file):
            try:
                data = load_json(oracle_file)
                scores['primary'] = _extract_metric_value(data, primary_metric_candidates)
            except (json.JSONDecodeError, ValueError):
                pass

    # Secondary task metric (RACE or MMLU)
    secondary_file = os.path.join(base_path, 'pre_results.json')
    if os.path.exists(secondary_file):
        try:
            data = load_json(secondary_file)
            scores['secondary'] = _extract_from_json(data, task_key, secondary_metric_candidates)
        except (json.JSONDecodeError, ValueError):
            pass

    return scores


def get_topk_scores(config: Dict[str, Any], topk: float) -> Dict[str, List[float]]:
    """Extract scores for a specific top-k value across all runs."""
    base_path = config['path']
    primary_task_key = config.get('primary_task_key', config['gsm8k_key'])
    task_key = config['task_key']
    primary_metric_candidates = config.get('primary_metric_candidates', ['exact_match,strict-match'])
    primary_flex_metric_candidates = config.get('primary_flex_metric_candidates', ['exact_match,flexible-extract'])
    secondary_metric_candidates = config.get('secondary_metric_candidates', ['acc,none'])

    scores: Dict[str, List[float]] = {
        'primary': [],
        'primary_flex': [],
        'secondary': [],
    }

    for run in range(NUM_RUNS):
        has_primary_for_run = False

        # Primary task metrics from *_train_task files
        train_file = _find_matching_file(base_path, run, topk, "_train_task", ["Race", "RACE", "MMLU"])
        if train_file and os.path.exists(train_file):
            try:
                data = load_json(train_file)
                primary = _extract_from_json(data, primary_task_key, primary_metric_candidates)
                if primary is not None:
                    scores['primary'].append(primary)
                    has_primary_for_run = True

                if config.get('include_flex', True):
                    primary_flex = _extract_from_json(data, primary_task_key, primary_flex_metric_candidates)
                    if primary_flex is not None:
                        scores['primary_flex'].append(primary_flex)
            except (json.JSONDecodeError, ValueError):
                pass

        # Optional oracle metrics fallback for primary (e.g., CodeAlpaca)
        if (not has_primary_for_run) and config.get('oracle_metric_file_prefix'):
            task_prefixes = ["RACE", "Race", "MMLU", "mmlu"]
            oracle_file = _find_matching_file(
                base_path,
                run,
                topk,
                "",
                [f"{config['oracle_metric_file_prefix']}_{p}" for p in task_prefixes],
            )
            if oracle_file and os.path.exists(oracle_file):
                try:
                    data = load_json(oracle_file)
                    primary = _extract_metric_value(data, primary_metric_candidates)
                    if primary is not None:
                        scores['primary'].append(primary)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Secondary task metrics from task result files
        secondary_file = _find_matching_file(base_path, run, topk, "", ["Race", "RACE", "MMLU"])
        if secondary_file and os.path.exists(secondary_file):
            try:
                data = load_json(secondary_file)
                secondary = _extract_from_json(data, task_key, secondary_metric_candidates)
                if secondary is not None:
                    scores['secondary'].append(secondary)
            except (json.JSONDecodeError, ValueError):
                pass

    return scores


def format_score(values: List[float]) -> str:
    """Format a list of scores as 'mean ± std'."""
    if not values:
        return "-"
    
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) == 3 else "N/A"
    
    if std == "N/A":
        return f"${mean:.3f}_{{N/A}}$"
    return f"${mean:.3f}_{{{std:.2f}}}$"


def format_single_score(value: float) -> str:
    """Format a single score value."""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def generate_table():
    """Generate the complete accuracy table."""

    all_columns = []
    for _, config in DATASETS.items():
        all_columns.extend(get_dataset_columns(config))

    table_width = max(120, 18 * (1 + len(all_columns)))

    print("=" * table_width)
    print("Accuracy Table: mean ± std over 3 seeds for Llama-3.1-8B IT")
    print("=" * table_width)
    print()

    # Header row
    header = [r"\multicolumn{1}{l}{\multirow{2}{*}{Top-k}}"]
    for _, config in DATASETS.items():
        for col in get_dataset_columns(config):
            header.append(col['header'])

    print(" | ".join(f"{h:^15}" for h in header))
    print("-" * table_width)

    # Pre-train row
    pretrain_row = ["Pre-train"]
    for _, config in DATASETS.items():
        scores = get_pretrain_scores(config)
        for col in get_dataset_columns(config):
            pretrain_row.append(format_single_score(scores.get(col['score_key'])))

    print(" | ".join(f"{v:^15}" for v in pretrain_row))
    print("-" * table_width)

    # Top-k rows
    for topk in TOP_K_VALUES:
        row = [f"${topk}$"]

        for _, config in DATASETS.items():
            scores = get_topk_scores(config, topk)
            for col in get_dataset_columns(config):
                row.append(format_score(scores[col['score_key']]))

        print(" | ".join(f"{v:^15}" for v in row))

    print("=" * table_width)


def generate_csv(output_file: str = "accuracy_table.csv"):
    """Generate CSV file with the table data."""
    import csv
    
    # Prepare data
    rows = []
    
    # Header
    header = ["Top-k%"]
    for _, config in DATASETS.items():
        for col in get_dataset_columns(config):
            header.append(col['header'])
    rows.append(header)
    
    # Pre-train row
    pretrain_row = ["Pre-train"]
    for _, config in DATASETS.items():
        scores = get_pretrain_scores(config)
        for col in get_dataset_columns(config):
            pretrain_row.append(format_single_score(scores.get(col['score_key'])))
    rows.append(pretrain_row)
    
    # Top-k rows
    for topk in TOP_K_VALUES:
        row = [str(topk)]

        for _, config in DATASETS.items():
            scores = get_topk_scores(config, topk)
            for col in get_dataset_columns(config):
                row.append(format_score(scores[col['score_key']]))

        rows.append(row)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"\nCSV file saved to: {output_file}")


def generate_latex_table(output_file: str = "accuracy_table.tex"):
    """Generate LaTeX table code."""
    
    lines = []
    lines.append("\\begin{table*}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Accuracy (mean $\\pm$ std over 3 seeds) for Llama-3.1-8B IT}")
    lines.append("\\label{tab:accuracy}")
    
    # Table structure
    dataset_columns = [get_dataset_columns(config) for config in DATASETS.values()]
    num_cols = 1 + sum(len(cols) for cols in dataset_columns)
    col_spec = "l" + "".join(" c" * len(cols) for cols in dataset_columns)
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header_parts = [r"\multicolumn{1}{l}{\multirow{2}{*}{Top-k}}"]
    for config, cols in zip(DATASETS.values(), dataset_columns):
        label = config['label']
        header_parts.append(f"\\multicolumn{{{len(cols)}}}{{c}}{{{label}}}")
    lines.append(" & ".join(header_parts) + " \\\\")

    cmidrules = []
    start_col = 2
    for cols in dataset_columns:
        end_col = start_col + len(cols) - 1
        cmidrules.append(f"\\cmidrule(rl){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    lines.append(" ".join(cmidrules))
    
    # Sub-header
    subheader_parts = [""]
    for cols in dataset_columns:
        for col in cols:
            subheader_parts.append(col['latex_subheader'])
    lines.append(" & ".join(subheader_parts) + " \\\\")
    lines.append("\\hline")

    # Pre-train row
    pretrain_parts = ["Pre-train"]
    for config, cols in zip(DATASETS.values(), dataset_columns):
        scores = get_pretrain_scores(config)
        for col in cols:
            pretrain_parts.append(format_single_score(scores.get(col['score_key'])))
    lines.append(" & ".join(pretrain_parts) + " \\\\")
    lines.append("\\hline")

    # Top-k rows
    for topk in TOP_K_VALUES:
        row_parts = [str(topk)]

        for config, cols in zip(DATASETS.values(), dataset_columns):
            scores = get_topk_scores(config, topk)
            for col in cols:
                row_parts.append(format_score(scores[col['score_key']]))

        lines.append(" & ".join(row_parts) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX table saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate accuracy table from evaluation results')
    parser.add_argument('--format', choices=['console', 'csv', 'latex', 'all'], default='console',
                        help='Output format (default: console)')
    parser.add_argument('--csv-output', default='accuracy_table.csv',
                        help='CSV output filename (default: accuracy_table.csv)')
    parser.add_argument('--latex-output', default='accuracy_table.tex',
                        help='LaTeX output filename (default: accuracy_table.tex)')
    
    args = parser.parse_args()
    
    if args.format in ['console', 'all']:
        generate_table()
    
    if args.format in ['csv', 'all']:
        generate_csv(args.csv_output)
    
    if args.format in ['latex', 'all']:
        generate_latex_table(args.latex_output)

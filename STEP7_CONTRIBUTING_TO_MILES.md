# Step 7: Contributing to Miles

**Welcome to the final step!** This guide will teach you how to navigate the Miles codebase, understand its architecture, and make meaningful contributions.

---

## Table of Contents

1. [Codebase Navigation](#1-codebase-navigation)
2. [Development Environment Setup](#2-development-environment-setup)
3. [Testing Structure](#3-testing-structure)
4. [CI/CD Pipeline](#4-cicd-pipeline)
5. [Common Contribution Areas](#5-common-contribution-areas)
6. [Best Practices](#6-best-practices)
7. [Making Your First Contribution](#7-making-your-first-contribution)

---

## 1. Codebase Navigation

### 1.1 Repository Structure

```
miles/
├── miles/                      # Core framework code
│   ├── backends/              # Training backends (FSDP, Megatron, SGLang)
│   ├── ray/                   # Ray orchestration and distributed coordination
│   ├── rollout/               # Generation and reward computation
│   ├── router/                # Routing logic for MoE models
│   └── utils/                 # Shared utilities
├── miles_plugins/             # Plugin system for custom integrations
│   ├── mbridge/              # Model architecture adapters
│   ├── megatron_bridge/      # Megatron-Core integration
│   └── models/               # Model-specific configurations
├── scripts/                   # Training scripts and configurations
│   ├── models/               # Model definition scripts
│   └── run-*.sh             # Example training scripts
├── tests/                     # Test suite
│   └── ci/                   # CI-specific utilities
├── docs/                      # Documentation
│   └── en/                   # English documentation
├── train.py                   # Main training entry point
└── train_async.py            # Asynchronous training variant
```

### 1.2 Core Modules Explained

#### **miles/backends/**
Handles the actual model training with different backends:

- **`fsdp_utils/`**: FSDP (Fully Sharded Data Parallel) implementation
  - `actor.py`: Main FSDP training actor with forward/backward/optimizer logic (lines 531-766)
  - `data_packing.py`: Efficient sequence packing with cu_seqlens (lines 11-105)
  - `training_utils.py`: Training loop helpers

- **`megatron_utils/`**: Megatron-Core integration for tensor/pipeline parallelism

- **`sglang_utils/`**: SGLang backend for efficient generation

**Key files to understand:**
- `miles/backends/fsdp_utils/actor.py:531-766` - Complete training loop
- `miles/backends/fsdp_utils/data_packing.py:11-105` - Data packing implementation

#### **miles/ray/**
Ray-based distributed orchestration:

- `rollout.py`: RolloutManager that coordinates generation across GPUs
- `actor_group.py`: RayTrainGroup for managing training actors
- `placement_group.py`: GPU allocation strategies
- `train_actor.py`: Base class for training actors

**Key concepts:**
- Actor model for distributed computing
- Placement groups for GPU topology-aware scheduling
- Async communication between rollout and training

#### **miles/rollout/**
Generation and reward computation:

- `sglang_rollout.py`: SGLang-based generation (lines 90-199 show sample creation)
- `rm_hub/`: Reward model implementations
  - `__init__.py`: Reward model registry and dispatcher
  - `math_dapo_utils.py`: Math problem rewards
  - `f1.py`: F1 score rewards
  - `gpqa.py`, `deepscaler.py`, etc.
- `filter_hub/`: Filtering strategies for samples
- `generate_hub/`: Custom generation strategies

**Adding a new reward model? Start here!**

#### **miles/utils/**
Shared utilities (20+ modules):

- `arguments.py`: Command-line argument definitions (350+ lines)
- `ppo_utils.py`: PPO and GAE math (lines 309-370: GAE, lines 125-148: PPO loss)
- `types.py`: Core data structures (`Sample`, `Status`, etc.)
- `data.py`: Data loading and preprocessing
- `distributed_utils.py`: Distributed training helpers
- `logging_utils.py`: Logging and metrics
- `memory_utils.py`: Memory management

**Most frequently modified:**
- `arguments.py` - adding new hyperparameters
- `ppo_utils.py` - modifying RL algorithms
- `types.py` - changing data structures

#### **miles_plugins/**
Plugin system for extending Miles without modifying core:

- `mbridge/`: Model adapters (e.g., `glm4moe.py` for GLM-4 MoE)
- `megatron_bridge/`: Megatron architecture bridges
- `models/`: Model-specific configurations

**Want to add a new model architecture? Use the plugin system!**

### 1.3 How to Find Things

#### Finding where a feature is implemented:

**Use case: "Where is speculative decoding implemented?"**

```bash
# Search for the keyword in code
grep -r "speculative" miles/ --include="*.py" | grep -v test

# Common locations:
# - miles/rollout/sglang_rollout.py:177-182 (tracking spec decode info)
# - docs/en/advanced/speculative-decoding.md (documentation)
```

**Use case: "Where are command-line arguments defined?"**

```bash
# All arguments are in one place
less miles/utils/arguments.py
# Search for specific arg: /--your-arg-name
```

**Use case: "How does GAE work?"**

```bash
# Implementation location
less miles/utils/ppo_utils.py
# Jump to line 309: :309
# See our detailed explanation: STEP4_RL_CONCEPTS.md
```

#### Key file reference by topic:

| Topic | Primary Files |
|-------|--------------|
| **Training loop** | `miles/backends/fsdp_utils/actor.py:531-766` |
| **PPO/GAE** | `miles/utils/ppo_utils.py:309-370, 125-148` |
| **Generation** | `miles/rollout/sglang_rollout.py:90-199` |
| **Reward models** | `miles/rollout/rm_hub/__init__.py` |
| **Data packing** | `miles/backends/fsdp_utils/data_packing.py:11-105` |
| **Ray orchestration** | `miles/ray/rollout.py`, `miles/ray/actor_group.py` |
| **Arguments** | `miles/utils/arguments.py` |
| **Sample structure** | `miles/utils/types.py` |

### 1.4 Understanding the Three-Tier Architecture

From the README, Miles uses a three-tier architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Ray Orchestrator                      │  (Tier 1)
│  - Coordinates rollout and training                     │
│  - Manages GPU allocation via placement groups          │
│  - Handles sample lifecycle (PENDING → COMPLETED)       │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
┌─────────▼──────────┐         ┌─────────▼──────────┐
│  Rollout Workers   │         │  Training Workers   │  (Tier 2)
│  (SGLang)          │         │  (FSDP/Megatron)   │
│  - Generate        │         │  - Forward pass    │
│  - Compute rewards │         │  - Compute loss    │
│  - Filter samples  │         │  - Backprop        │
└────────────────────┘         └────────────────────┘
          │                               │
          └───────────────┬───────────────┘
                          │
                ┌─────────▼──────────┐
                │   Data Pipeline    │              (Tier 3)
                │  - Pack sequences  │
                │  - Mask tokens     │
                │  - Create batches  │
                └────────────────────┘
```

**Code locations:**
- **Tier 1**: `miles/ray/rollout.py`, `miles/ray/actor_group.py`
- **Tier 2**:
  - Rollout: `miles/rollout/sglang_rollout.py`
  - Training: `miles/backends/fsdp_utils/actor.py`
- **Tier 3**: `miles/backends/fsdp_utils/data_packing.py`

---

## 2. Development Environment Setup

### 2.1 Initial Setup

```bash
# 1. Clone the repository
git clone https://github.com/radixark/miles.git
cd miles

# 2. Install in development mode
pip install -e .

# 3. Install development dependencies
pip install pre-commit pytest black isort ruff

# 4. Set up pre-commit hooks
pre-commit install
```

### 2.2 Pre-commit Hooks

Miles uses pre-commit hooks to enforce code quality. Every commit automatically runs:

**Configured in `.pre-commit-config.yaml`:**

1. **Ruff** (linting): Catches bugs, unused imports, undefined names
2. **Autoflake**: Removes unused imports automatically
3. **isort**: Sorts imports (Black-compatible style)
4. **Black**: Code formatting (line length: 119)
5. **Standard checks**: YAML validation, private key detection, large file prevention

**What this means for you:**
- Your code will be automatically formatted on commit
- Linting errors will block commits (fix them first!)
- Import order will be corrected automatically

**Example workflow:**

```bash
# Make your changes
vim miles/utils/my_new_feature.py

# Try to commit
git add miles/utils/my_new_feature.py
git commit -m "Add new feature"

# Pre-commit hooks run automatically
# If they fail, fix the issues and recommit
# If they pass, your commit succeeds!
```

**Manual pre-commit run:**

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files miles/utils/my_file.py
```

### 2.3 Code Style Configuration

From `pyproject.toml`:

#### Black (formatting):
```toml
[tool.black]
line_length = 119
```

#### isort (import sorting):
```toml
[tool.isort]
profile = "black"
line_length = 119
py_version = 310
known_first_party = ["miles", "miles_plugins"]
known_third_party = ["megatron", "wandb", "ray", "transformers"]
```

**Import order example:**
```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import asyncio
import os
from typing import Any

# 3. Third-party
import ray
import torch
from transformers import AutoModelForCausalLM

# 4. First-party (miles)
from miles.utils.types import Sample
from miles.backends.fsdp_utils.actor import FSDPActor

# 5. Local/relative
from .my_module import my_function
```

#### Ruff (linting):
```toml
[tool.ruff]
line-length = 320  # Very permissive for complex expressions
select = [
    "E",   # Pycodestyle errors (indentation, whitespace)
    "F",   # Pyflakes (unused imports, undefined names)
    "B",   # Bugbear (common bugs)
    "UP",  # pyupgrade (modernization)
]
ignore = [
    "E402",  # module-import-not-at-top-of-file (allowed for conditional imports)
]
```

---

## 3. Testing Structure

### 3.1 Test Organization

```
tests/
├── test_qwen2.5_0.5B_gsm8k.py           # End-to-end training test
├── test_qwen2.5_0.5B_gsm8k_async.py     # Async training variant
├── test_qwen3_0.6B_fsdp_colocated_2xGPU.py  # Colocated mode test
├── test_qwen3_0.6B_fsdp_distributed.py      # Distributed mode test
├── test_qwen3_30B_A3B.py                    # Large MoE model test
├── test_quick_start_glm4_9B.py              # Quick start verification
├── test_external_rollout.py                 # External rollout test
└── ci/
    └── gpu_lock_exec.py                     # GPU locking for CI
```

### 3.2 Test Structure Pattern

All tests follow this pattern:

```python
import miles.utils.external_utils.command_utils as U

# Configuration
MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 2

def prepare():
    """Download models and datasets"""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")

def execute():
    """Run the training"""
    # Define argument groups
    ckpt_args = "--hf-checkpoint /root/models/... --ref-load /root/models/..."
    rollout_args = "--prompt-data ... --rollout-batch-size 32 ..."
    perf_args = "--tensor-model-parallel-size 1 --sequence-parallel ..."
    # ... more arg groups

    train_args = f"{ckpt_args} {rollout_args} {perf_args} ..."

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )

if __name__ == "__main__":
    prepare()
    execute()
```

**Key components:**

1. **Environment variables** for test configuration:
   ```python
   FEW_GPU = U.get_bool_env_var("MILES_TEST_FEW_GPU", "1")
   TIGHT_DEVICE_MEMORY = U.get_bool_env_var("MILES_TEST_TIGHT_DEVICE_MEMORY", "1")
   ```

2. **CI-specific arguments**:
   ```python
   ci_args = (
       "--ci-test "
       "--ci-disable-kl-checker "
       "--ci-metric-checker-key eval/gsm8k "
       "--ci-metric-checker-threshold 0.55 "
   )
   ```

3. **Utility helpers** from `command_utils`:
   - `U.exec_command()`: Run shell commands
   - `U.hf_download_dataset()`: Download HuggingFace datasets
   - `U.execute_train()`: Execute training with proper setup
   - `U.get_default_wandb_args()`: Get W&B configuration
   - `U.get_env_enable_infinite_run()`: Check infinite run mode

### 3.3 Writing a New Test

**Example: Adding a test for a new model**

```python
# tests/test_my_new_model.py

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "MyOrg/MyModel-7B"
MODEL_TYPE = "my-model-7B"
NUM_GPUS = 4

def prepare():
    """Download model and data"""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download {MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("my-dataset")

def execute():
    """Run training"""
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}/"

    rollout_args = (
        "--prompt-data /root/datasets/my-dataset/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 512 "
        "--rm-type my-custom-reward "
        "--num-rollout 100 "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
        "--sequence-parallel "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
    )

    ppo_args = (
        "--advantage-estimator gae "
        "--gae-lambda 0.95 "
        "--gamma 1.0 "
        "--eps-clip 0.2 "
        "--use-kl-loss "
        "--kl-loss-coef 0.02 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 5e-7 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
    )

    ci_args = (
        "--ci-test "
        "--ci-metric-checker-key eval/my_metric "
        "--ci-metric-checker-threshold 0.6 "
    )

    train_args = f"{ckpt_args} {rollout_args} {perf_args} {ppo_args} {optimizer_args} {ci_args}"

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )

if __name__ == "__main__":
    prepare()
    execute()
```

### 3.4 Running Tests Locally

```bash
# Run a specific test
python tests/test_qwen2.5_0.5B_gsm8k.py

# Run with pytest
pytest tests/test_qwen2.5_0.5B_gsm8k.py -v

# Run all tests (requires GPUs!)
pytest tests/ -v

# Run with specific markers (from pyproject.toml)
pytest -m unit tests/          # Only unit tests
pytest -m "not integration" tests/  # Skip integration tests
```

**Environment variables for testing:**

```bash
# Use fewer GPUs for testing
export MILES_TEST_FEW_GPU=1

# Tight device memory (lower VRAM usage)
export MILES_TEST_TIGHT_DEVICE_MEMORY=1

# Enable infinite run (for debugging)
export MILES_TEST_ENABLE_INFINITE_RUN=true
```

### 3.5 Test Markers

From `pyproject.toml`, Miles supports these pytest markers:

```python
# In your test file
import pytest

@pytest.mark.unit
def test_gae_computation():
    """Unit test for GAE calculation"""
    pass

@pytest.mark.integration
def test_rollout_training_integration():
    """Integration test for rollout + training"""
    pass

@pytest.mark.system
def test_full_training_pipeline():
    """System test for complete training"""
    pass

@pytest.mark.skipduringci
def test_expensive_operation():
    """Skip in CI, run locally"""
    pass
```

Run specific markers:
```bash
pytest -m unit tests/              # Only unit tests
pytest -m "integration or system"  # Integration or system tests
pytest -m "not skipduringci"       # Skip expensive tests
```

---

## 4. CI/CD Pipeline

### 4.1 GitHub Actions Workflows

Miles has two main CI workflows:

#### **Workflow 1: Pre-commit** (`.github/workflows/pre-commit.yml`)

**Triggered on:**
- Every push to `main`
- Every PR opened/synchronized/reopened

**What it does:**
1. Checks out code
2. Sets up Python 3.10
3. Installs pre-commit
4. Runs all pre-commit hooks on all files

**Configuration:**
```yaml
- name: Run pre-commit on all files
  run: pre-commit run --all-files --show-diff-on-failure --color=always
```

**This ensures:**
- Code is formatted with Black
- Imports are sorted with isort
- Linting passes with Ruff
- No large files, private keys, or YAML errors

**If this fails:** Fix the linting/formatting issues locally and push again.

#### **Workflow 2: PR Test** (`.github/workflows/pr-test.yml`)

**Triggered on:**
- PRs to `main` with labels `run-ci-short` or `run-ci-long`
- Manual workflow dispatch

**Two test suites:**

1. **e2e-test-short** (requires label: `run-ci-short`)
   - Tests: `test_quick_start_glm4_9B.py`, `test_qwen3_30B_A3B.py`
   - GPUs: 8 per test
   - Docker image: `radixark/miles:latest`
   - Purpose: Quick smoke tests for large models

2. **e2e-test-long** (requires label: `run-ci-long`)
   - Tests: 4 comprehensive tests (GSM8K, FSDP variants)
   - GPUs: 2 per test
   - Purpose: Full training verification

**CI Environment:**
```yaml
container:
  image: radixark/miles:latest
  options: >
    --gpus all
    --ipc=host
    --shm-size=16g
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v /data/miles_ci:/data/miles_ci
    -v /data/miles_ci/models:/root/models
    -v /data/miles_ci/datasets:/root/datasets
```

**GPU locking mechanism:**
The CI uses `tests/ci/gpu_lock_exec.py` to prevent GPU conflicts:

```bash
python tests/ci/gpu_lock_exec.py --count 2 -- python tests/test_qwen2.5_0.5B_gsm8k.py
```

This ensures that even with multiple tests running in parallel, each test gets exclusive GPU access.

### 4.2 Adding CI Tests to Your PR

**Step 1:** Push your changes to a branch

```bash
git checkout -b feature/my-new-feature
git add .
git commit -m "Add my feature"
git push origin feature/my-new-feature
```

**Step 2:** Create a PR on GitHub

**Step 3:** Add the appropriate label to trigger CI:
- `run-ci-short`: For quick validation (8 GPU tests, ~10-15 minutes)
- `run-ci-long`: For comprehensive validation (2 GPU tests, ~30-60 minutes)

**Step 4:** Monitor the CI run in the "Actions" tab

**Step 5:** If tests fail:
1. Check the logs in GitHub Actions
2. Reproduce locally: `python tests/test_that_failed.py`
3. Fix the issue
4. Push the fix (CI will re-run automatically)

### 4.3 CI Success Criteria

Your PR CI passes when:

1. **Pre-commit checks pass:**
   - Black formatting ✓
   - isort import sorting ✓
   - Ruff linting ✓
   - No large files or private keys ✓

2. **Training tests pass:**
   - Training runs without crashes ✓
   - Metrics reach target thresholds ✓
   - No memory errors ✓

**Example CI metric checker:**
```python
ci_args = (
    "--ci-test "
    "--ci-metric-checker-key eval/gsm8k "
    "--ci-metric-checker-threshold 0.55 "
)
```

This means the test passes only if `eval/gsm8k` metric >= 0.55.

---

## 5. Common Contribution Areas

### 5.1 Adding New Model Support

**Goal:** Add support for a new model architecture (e.g., Llama-4, GPT-5, etc.)

#### **Option 1: Using the Plugin System** (Recommended)

**Step-by-step:**

1. **Create a model adapter plugin:**

```python
# miles_plugins/mbridge/my_new_model.py

from miles_plugins.mbridge.base import MegatronBridgeBase

class MyNewModelBridge(MegatronBridgeBase):
    """Bridge for MyNewModel architecture"""

    def __init__(self, args):
        super().__init__(args)
        # Model-specific initialization
        self.num_experts = args.num_experts
        self.expert_parallel_size = args.expert_model_parallel_size

    def build_model(self, **kwargs):
        """Build the model architecture"""
        # Import your model class
        from my_model_lib import MyNewModel

        model = MyNewModel(
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers,
            num_attention_heads=self.args.num_attention_heads,
            # ... other args
        )
        return model

    def load_weights(self, model, checkpoint_path):
        """Load HuggingFace weights"""
        # Convert HF weights to Miles format
        # See miles_plugins/mbridge/glm4moe.py for example
        pass

    def get_model_config(self):
        """Return model configuration"""
        return {
            "vocab_size": self.args.padded_vocab_size,
            "hidden_size": self.args.hidden_size,
            "num_layers": self.args.num_layers,
            # ... other config
        }
```

2. **Register your bridge:**

```python
# miles_plugins/mbridge/__init__.py

from .my_new_model import MyNewModelBridge

BRIDGE_REGISTRY = {
    "glm4moe": "miles_plugins.mbridge.glm4moe.GLM4MOEBridge",
    "my-new-model": "miles_plugins.mbridge.my_new_model.MyNewModelBridge",  # Add this
}
```

3. **Create a model configuration script:**

```bash
# scripts/models/my-new-model-7B.sh

export MODEL_SIZE=7B
export NUM_LAYERS=32
export HIDDEN_SIZE=4096
export NUM_ATTN_HEADS=32
export NUM_KV_HEADS=8
export SEQ_LENGTH=8192

# MoE-specific (if applicable)
export NUM_EXPERTS=8
export NUM_EXPERTS_PER_TOK=2

# Parallelism
export TP=2  # Tensor parallel
export PP=1  # Pipeline parallel
export EP=1  # Expert parallel

# Add model-specific args
export EXTRA_ARGS="
    --use-rotary-position-embeddings
    --use-my-custom-attention
    --activation gelu
"
```

4. **Create a training script:**

```bash
# scripts/run-my-new-model-7B.sh

#!/bin/bash

# Load model config
source scripts/models/my-new-model-7B.sh

# Training configuration
python train.py \
    --hf-checkpoint /path/to/model \
    --ref-load /path/to/model \
    --megatron-to-hf-mode bridge \
    --megatron-model-type my-new-model-7B \
    \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTN_HEADS \
    --num-key-value-heads $NUM_KV_HEADS \
    --seq-length $SEQ_LENGTH \
    \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --expert-model-parallel-size $EP \
    \
    --prompt-data /data/my_dataset.parquet \
    --rm-type my-reward \
    $EXTRA_ARGS
```

5. **Test your integration:**

```python
# tests/test_my_new_model_7B.py

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "MyOrg/MyNewModel-7B"
MODEL_TYPE = "my-new-model-7B"

def prepare():
    U.exec_command("mkdir -p /root/models")
    U.exec_command(f"huggingface-cli download {MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")

def execute():
    # Use your training script
    U.exec_command("bash scripts/run-my-new-model-7B.sh --ci-test --num-rollout 10")

if __name__ == "__main__":
    prepare()
    execute()
```

#### **Option 2: Direct Integration** (For core models)

If your model should be part of the core Miles framework:

1. **Add model architecture to Megatron bridge:**
   - Modify `miles/utils/megatron_bridge_utils.py`
   - Add model-specific layer definitions

2. **Update argument parser:**
   - Modify `miles/utils/arguments.py`
   - Add model-specific arguments

3. **Add weight conversion logic:**
   - Create converter in `miles/backends/megatron_utils/`
   - Handle HF → Megatron weight mapping

**Example contribution areas for new models:**
- [ ] Model architecture adapter (plugin or core)
- [ ] Weight conversion from HuggingFace
- [ ] Configuration scripts
- [ ] Example training script
- [ ] CI test
- [ ] Documentation (add to `docs/en/examples/`)

---

### 5.2 Adding New Reward Model Types

**Goal:** Add a custom reward function for your task

**Location:** `miles/rollout/rm_hub/`

#### **Step 1: Implement your reward function**

```python
# miles/rollout/rm_hub/my_custom_reward.py

from miles.utils.types import Sample

async def compute_my_custom_reward(args, sample: Sample) -> float:
    """
    Custom reward function.

    Args:
        args: Training arguments (access via args.my_custom_param)
        sample: Sample object with prompt, response, label, metadata

    Returns:
        float: Reward score (typically 0.0 to 1.0)
    """
    response = sample.response
    label = sample.label

    # Example: Check if response contains specific keywords
    keywords = ["correct", "accurate", "precise"]
    keyword_count = sum(1 for kw in keywords if kw in response.lower())

    # Example: Compare with ground truth
    if label is not None:
        similarity = compute_similarity(response, label)
        return 0.5 * similarity + 0.5 * (keyword_count / len(keywords))

    return keyword_count / len(keywords)

def compute_similarity(text1: str, text2: str) -> float:
    """Helper function to compute text similarity"""
    # Implement your similarity metric
    # Could use: edit distance, BLEU, ROUGE, embedding similarity, etc.
    pass
```

#### **Step 2: Register in the reward model hub**

```python
# miles/rollout/rm_hub/__init__.py

from .my_custom_reward import compute_my_custom_reward

async def async_rm(args, sample: Sample, **kwargs):
    """Main reward model dispatcher"""

    # ... existing code ...

    # Add your reward type
    elif rm_type == "my-custom":
        return await compute_my_custom_reward(args, sample)

    # ... rest of function ...
```

#### **Step 3: Add command-line arguments (if needed)**

```python
# miles/utils/arguments.py

def add_rollout_args(parser: FlexibleArgumentParser):
    # ... existing args ...

    group.add_argument(
        "--my-custom-reward-threshold",
        type=float,
        default=0.5,
        help="Threshold for my custom reward",
    )
```

#### **Step 4: Use in training**

```bash
python train.py \
    --rm-type my-custom \
    --my-custom-reward-threshold 0.7 \
    --prompt-data /data/my_dataset.parquet \
    # ... other args
```

#### **Real-world examples in the codebase:**

**Math reward** (`miles/rollout/rm_hub/math_dapo_utils.py`):
```python
def compute_score(response: str, label: str) -> float:
    """Compute reward for math problems"""
    predicted = extract_answer(response)
    if predicted is None:
        return 0.0

    ground_truth = label

    # Normalize answers
    pred_norm = normalize_answer(predicted)
    truth_norm = normalize_answer(ground_truth)

    # Binary reward
    return 1.0 if pred_norm == truth_norm else 0.0
```

**F1 score reward** (`miles/rollout/rm_hub/f1.py`):
```python
async def f1_score(args, sample: Sample):
    """F1 score for classification tasks"""
    response = sample.response.strip()
    label = sample.label.strip()

    # Tokenize
    response_tokens = set(response.split())
    label_tokens = set(label.split())

    # Compute F1
    if len(response_tokens) == 0 or len(label_tokens) == 0:
        return 0.0

    intersection = response_tokens & label_tokens
    precision = len(intersection) / len(response_tokens)
    recall = len(intersection) / len(label_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

**Remote reward model** (for API-based rewards):
```python
async def remote_rm(args, sample: Sample):
    """Call external reward model API"""
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()
```

**Custom reward via command line:**
```bash
# No code changes needed! Just specify a path
python train.py \
    --custom-rm-path my_module.my_reward_function \
    --prompt-data /data/dataset.parquet \
    # ... other args
```

---

### 5.3 Performance Optimizations

**Goal:** Make training faster or more memory-efficient

#### **Common optimization areas:**

1. **Memory optimization:**
   - **Location:** `miles/backends/fsdp_utils/actor.py`
   - **What to improve:**
     - Activation checkpointing strategies (lines 295-318: sleep/wake_up)
     - Gradient accumulation logic
     - FSDP sharding configuration

2. **Data packing optimization:**
   - **Location:** `miles/backends/fsdp_utils/data_packing.py`
   - **What to improve:**
     - Sequence packing algorithm (lines 11-105)
     - cu_seqlens computation
     - Batch construction

3. **Generation speed:**
   - **Location:** `miles/rollout/sglang_rollout.py`
   - **What to improve:**
     - Speculative decoding integration (lines 177-182)
     - Batch size tuning
     - Temperature/sampling strategies

4. **Communication optimization:**
   - **Location:** `miles/backends/fsdp_utils/actor.py:531-766`
   - **What to improve:**
     - Gradient accumulation (reduce communication)
     - Async communication patterns
     - Weight synchronization

#### **Example: Optimizing gradient accumulation**

**Current implementation:**
```python
# miles/backends/fsdp_utils/actor.py (simplified)

for microbatch in microbatches:
    loss = forward_backward(microbatch)
    accumulated_loss += loss

# Synchronize gradients (expensive!)
all_reduce_gradients()

optimizer.step()
```

**Optimization: Overlap communication with computation**
```python
# Proposed improvement

with torch.cuda.stream(compute_stream):
    for i, microbatch in enumerate(microbatches):
        loss = forward_backward(microbatch)
        accumulated_loss += loss

        # Overlap gradient all-reduce with next forward pass
        if i < len(microbatches) - 1:
            with torch.cuda.stream(comm_stream):
                async_all_reduce_gradients()

optimizer.step()
```

**How to contribute this:**
1. Benchmark current performance
2. Implement optimization
3. Benchmark new performance
4. Document speedup in PR (e.g., "15% faster training")
5. Add test to verify correctness

#### **Example: Memory optimization for large models**

**Problem:** 70B model with FSDP runs out of memory

**Current approach:**
```python
# All parameters in GPU memory
model = build_model()
model = FSDP(model)
```

**Optimization: CPU offloading**
```python
# Offload to CPU when not in use
model = build_model()

cpu_offload_config = CPUOffloadConfig(
    offload_params=True,
    offload_optimizer=True,
)

model = FSDP(
    model,
    cpu_offload=cpu_offload_config,
)
```

**Where to implement:**
- `miles/backends/fsdp_utils/actor.py:295-318` - sleep/wake_up methods
- `miles/utils/memory_utils.py` - memory management helpers

---

### 5.4 Bug Fixes

**Goal:** Fix a bug in the codebase

#### **Typical bug hunting workflow:**

1. **Reproduce the bug:**
   ```bash
   # Run the failing test or command
   python tests/test_that_fails.py

   # Or reproduce manually
   python train.py --your-failing-config
   ```

2. **Narrow down the location:**
   ```bash
   # Add debug prints
   # In miles/backends/fsdp_utils/actor.py

   def _train_step(self, batch):
       print(f"DEBUG: batch shape = {batch.shape}")  # Add this
       loss = self.forward(batch)
       print(f"DEBUG: loss = {loss}")  # And this
       return loss
   ```

3. **Use git bisect to find when it broke:**
   ```bash
   git bisect start
   git bisect bad  # Current commit is bad
   git bisect good v0.2.0  # v0.2.0 was good
   # Git will checkout commits; test each one
   python tests/test_that_fails.py
   git bisect good  # or bad
   # Repeat until you find the breaking commit
   ```

4. **Fix the bug:**
   - Modify the relevant file
   - Add a test that would have caught this bug
   - Verify the fix works

5. **Submit a PR:**
   ```bash
   git checkout -b fix/descriptive-bug-name
   git add <fixed-files>
   git commit -m "Fix: <concise description of bug and fix>"
   git push origin fix/descriptive-bug-name
   ```

#### **Common bug patterns in RL training:**

**Bug 1: Shape mismatches**
- **Location:** Data packing, loss computation
- **Example:**
  ```python
  # Bug: Assuming all sequences have same length
  advantages = advantages.view(batch_size, seq_len)  # WRONG! Packed sequences have variable length

  # Fix: Use cu_seqlens
  advantages = unpack_sequences(advantages, cu_seqlens)
  ```

**Bug 2: Gradient accumulation errors**
- **Location:** `miles/backends/fsdp_utils/actor.py`
- **Example:**
  ```python
  # Bug: Not scaling loss by accumulation steps
  loss.backward()

  # Fix: Scale by number of microbatches
  (loss / num_accumulation_steps).backward()
  ```

**Bug 3: Reward computation edge cases**
- **Location:** `miles/rollout/rm_hub/`
- **Example:**
  ```python
  # Bug: Division by zero
  precision = tp / (tp + fp)  # Crashes if tp + fp == 0

  # Fix: Handle edge case
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  ```

**Bug 4: Memory leaks**
- **Location:** Any training loop
- **Example:**
  ```python
  # Bug: Accumulating tensors in a list
  losses = []
  for batch in batches:
      loss = compute_loss(batch)
      losses.append(loss)  # Keeps computation graph in memory!

  # Fix: Detach or convert to Python float
  losses = []
  for batch in batches:
      loss = compute_loss(batch)
      losses.append(loss.item())  # Just the scalar value
  ```

---

### 5.5 Documentation Improvements

**Goal:** Improve or add documentation

#### **Documentation structure:**

```
docs/
├── en/
│   ├── get_started/
│   │   ├── quick_start.md
│   │   ├── usage.md
│   │   └── qa.md
│   ├── examples/
│   │   ├── glm4-9B.md
│   │   ├── qwen3-4B.md
│   │   └── deepseek-r1.md
│   ├── advanced/
│   │   ├── speculative-decoding.md
│   │   ├── fault-tolerance.md
│   │   └── arch-support-beyond-megatron.md
│   └── developer_guide/
│       └── debug.md
└── README.md
```

#### **Adding documentation for a new feature:**

1. **Choose the right location:**
   - Tutorial/example → `docs/en/examples/`
   - Advanced feature → `docs/en/advanced/`
   - Troubleshooting → `docs/en/get_started/qa.md`
   - Developer guide → `docs/en/developer_guide/`

2. **Follow the template:**

```markdown
# Feature Name

## Overview

Brief description of what this feature does and why it's useful.

## Quick Start

Minimal working example:

\`\`\`bash
python train.py \
    --feature-flag \
    --feature-param 42 \
    # ... minimal args
\`\`\`

## Detailed Usage

### Configuration

Explain all configuration options:

- `--feature-flag`: Enable the feature
- `--feature-param`: Control feature behavior (default: 42)

### Example

Full working example with explanations:

\`\`\`bash
# Step 1: Prepare data
python prepare_data.py --input data.json

# Step 2: Run training
python train.py \
    --feature-flag \
    --feature-param 100 \
    --prompt-data data.parquet \
    # ... complete args

# Step 3: Evaluate
python evaluate.py --checkpoint output/
\`\`\`

### How It Works

Technical explanation of the implementation:

1. **Step 1**: What happens first
2. **Step 2**: What happens next
3. **Step 3**: Final result

**Code reference:** `miles/module/file.py:100-200`

## Best Practices

- **Do this**: Recommended approach
- **Don't do this**: Common pitfalls

## Troubleshooting

### Error: "Something went wrong"

**Cause:** Why this error happens

**Solution:** How to fix it

## Related Features

- [Other Feature](link.md)
- [Related Concept](link.md)
```

3. **Add code examples:**
   - Include runnable code snippets
   - Show expected output
   - Explain key parameters

4. **Link from main README:**
   ```markdown
   # In docs/README.md or main README.md

   - [My New Feature](docs/en/advanced/my-new-feature.md)
   ```

#### **Improving existing documentation:**

Common improvements needed:
- [ ] Add missing command-line arguments
- [ ] Update outdated examples
- [ ] Add troubleshooting sections
- [ ] Improve code comments in complex functions
- [ ] Add type hints
- [ ] Document edge cases

**Example: Adding docstrings**

```python
# Before (no docstring)
def compute_advantages(rewards, values, gamma, lambda_):
    advantages = []
    # ... code ...
    return advantages

# After (comprehensive docstring)
def compute_advantages(rewards, values, gamma, lambda_):
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE combines temporal difference errors using exponential weighting
    to balance bias and variance in advantage estimation.

    Args:
        rewards: List of rewards for each timestep (length T)
        values: List of value estimates for each timestep (length T+1)
        gamma: Discount factor for future rewards (typically 0.99)
        lambda_: GAE lambda parameter for bias-variance tradeoff (typically 0.95)

    Returns:
        advantages: List of advantage estimates for each timestep (length T)

    Formula:
        A[t] = δ[t] + γλ * A[t+1]
        where δ[t] = r[t] + γ*V[t+1] - V[t]

    Note:
        Must iterate backward from t=T-1 to t=0 due to dependency on A[t+1].

    Example:
        >>> rewards = [1.0, 0.5, 2.0]
        >>> values = [0.0, 0.5, 1.0, 0.0]
        >>> advantages = compute_advantages(rewards, values, gamma=0.99, lambda_=0.95)
        >>> print(advantages)
        [2.87, 1.85, 1.01]

    References:
        - Schulman et al. (2016): "High-Dimensional Continuous Control Using GAE"
        - Implementation: miles/utils/ppo_utils.py:309-370
    """
    advantages = []
    # ... code ...
    return advantages
```

---

## 6. Best Practices

### 6.1 Code Style Guidelines

#### **Formatting (enforced by pre-commit):**

✅ **DO:**
```python
# Black-formatted, 119 char line length
def train_model(
    model,
    data,
    learning_rate=1e-5,
    batch_size=32,
    num_epochs=10,
):
    """Train the model with given parameters."""
    for epoch in range(num_epochs):
        for batch in data:
            loss = model(batch)
            loss.backward()
```

❌ **DON'T:**
```python
# Inconsistent spacing, too long lines, missing docstring
def train_model(model,data,learning_rate=1e-5,batch_size=32,num_epochs=10):
    for epoch in range(num_epochs):
        for batch in data:
            loss=model(batch)
            loss.backward( )
```

#### **Imports (enforced by isort):**

✅ **DO:**
```python
# Properly ordered and grouped
from __future__ import annotations

import os
from typing import Any, Dict, List

import ray
import torch
from transformers import AutoTokenizer

from miles.utils.types import Sample
from miles.backends.fsdp_utils.actor import FSDPActor

from .local_module import helper_function
```

❌ **DON'T:**
```python
# Random order, mixed grouping
from .local_module import helper_function
from miles.utils.types import Sample
import torch
from transformers import AutoTokenizer
import os
import ray
from typing import Any, Dict, List
from miles.backends.fsdp_utils.actor import FSDPActor
```

#### **Naming conventions:**

✅ **DO:**
```python
# Clear, descriptive names
class FSDPTrainingActor:
    def __init__(self, model_config: dict):
        self.num_layers = model_config["num_layers"]
        self.hidden_size = model_config["hidden_size"]

    def compute_ppo_loss(self, log_probs, old_log_probs, advantages):
        ratio = (log_probs - old_log_probs).exp()
        return compute_clipped_loss(ratio, advantages)

def compute_generalized_advantage_estimation(
    rewards: list[float],
    values: list[float],
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> list[float]:
    """Compute GAE advantages."""
    pass
```

❌ **DON'T:**
```python
# Vague, abbreviated names
class Actor:
    def __init__(self, cfg: dict):
        self.nl = cfg["num_layers"]
        self.hs = cfg["hidden_size"]

    def loss(self, lp, olp, adv):
        r = (lp - olp).exp()
        return compute_clipped_loss(r, adv)

def gae(r: list[float], v: list[float], g: float = 0.99, l: float = 0.95) -> list[float]:
    """Compute GAE."""  # What's GAE? What are these params?
    pass
```

#### **Type hints:**

✅ **DO:**
```python
from typing import Optional
from miles.utils.types import Sample

async def compute_reward(
    args,
    sample: Sample,
    use_cache: bool = True,
) -> float:
    """Compute reward for a sample."""
    reward: float = 0.0
    # ... computation ...
    return reward

def pack_sequences(
    sequences: list[list[int]],
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Pack variable-length sequences."""
    packed_tokens: list[int] = []
    cu_seqlens: list[int] = [0]
    # ... packing logic ...
    return packed_tokens, cu_seqlens
```

❌ **DON'T:**
```python
# No type hints
async def compute_reward(args, sample, use_cache=True):
    reward = 0.0
    return reward

def pack_sequences(sequences, max_length):
    packed_tokens = []
    cu_seqlens = [0]
    return packed_tokens, cu_seqlens
```

### 6.2 Testing Requirements

#### **Every contribution should include tests:**

✅ **DO:**
```python
# New feature: Custom reward model
# File: miles/rollout/rm_hub/my_reward.py

async def my_custom_reward(args, sample: Sample) -> float:
    """My custom reward function."""
    # ... implementation ...
    pass

# File: tests/test_my_reward.py

import pytest
from miles.utils.types import Sample, Status
from miles.rollout.rm_hub.my_reward import my_custom_reward

@pytest.mark.unit
async def test_my_custom_reward_basic():
    """Test basic reward computation."""
    sample = Sample(
        prompt="Question",
        response="Answer",
        label="Correct Answer",
        status=Status.COMPLETED,
    )

    class Args:
        pass

    args = Args()
    reward = await my_custom_reward(args, sample)

    assert isinstance(reward, float)
    assert 0.0 <= reward <= 1.0

@pytest.mark.unit
async def test_my_custom_reward_edge_cases():
    """Test edge cases."""
    # Empty response
    sample = Sample(response="", label="Something")
    reward = await my_custom_reward(Args(), sample)
    assert reward == 0.0

    # Perfect match
    sample = Sample(response="Exact", label="Exact")
    reward = await my_custom_reward(Args(), sample)
    assert reward == 1.0
```

❌ **DON'T:**
```python
# Contribution without tests
# Just submitting miles/rollout/rm_hub/my_reward.py
# No way to verify it works!
```

#### **Test coverage expectations:**

For new features, aim for:
- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test interaction between components
- **End-to-end test**: Test complete training pipeline (if applicable)

**Example test structure for a new feature:**

```python
# tests/test_new_feature.py

import pytest

# Unit tests (fast, isolated)
@pytest.mark.unit
def test_feature_basic():
    """Test basic functionality."""
    pass

@pytest.mark.unit
def test_feature_edge_cases():
    """Test edge cases."""
    pass

# Integration tests (medium speed)
@pytest.mark.integration
def test_feature_with_dataloader():
    """Test feature integrated with data loading."""
    pass

@pytest.mark.integration
def test_feature_with_model():
    """Test feature integrated with model."""
    pass

# System tests (slow, full pipeline)
@pytest.mark.system
def test_feature_full_training():
    """Test feature in complete training run."""
    pass
```

Run specific test levels:
```bash
pytest -m unit tests/             # Fast unit tests only
pytest -m integration tests/      # Integration tests
pytest -m system tests/           # Full system tests
pytest tests/                     # All tests
```

### 6.3 Documentation Standards

#### **Code documentation:**

Every public function/class should have:
1. **Docstring** explaining what it does
2. **Args** section describing parameters
3. **Returns** section describing output
4. **Example** showing usage (if non-trivial)

✅ **DO:**
```python
def pack_sequences(
    sequences: list[list[int]],
    max_length: int,
    padding_token: int = 0,
) -> tuple[list[int], list[int]]:
    """
    Pack variable-length sequences into a single sequence.

    This eliminates padding waste by concatenating sequences and tracking
    boundaries with cu_seqlens (cumulative sequence lengths).

    Args:
        sequences: List of token sequences, each with variable length
        max_length: Maximum total length of packed sequence
        padding_token: Token ID for padding (default: 0)

    Returns:
        packed_tokens: Concatenated token sequences
        cu_seqlens: Cumulative sequence lengths [0, len(seq1), len(seq1)+len(seq2), ...]

    Example:
        >>> sequences = [[1, 2], [3, 4, 5], [6]]
        >>> packed, cu = pack_sequences(sequences, max_length=10)
        >>> print(packed)
        [1, 2, 3, 4, 5, 6]
        >>> print(cu)
        [0, 2, 5, 6]

    Note:
        cu_seqlens[i:i+2] gives the start and end indices for sequence i.

    References:
        - Flash Attention 2: github.com/Dao-AILab/flash-attention
        - Data packing explanation: miles/data_packing_explained.md
    """
    packed_tokens = []
    cu_seqlens = [0]
    # ... implementation ...
    return packed_tokens, cu_seqlens
```

❌ **DON'T:**
```python
def pack_sequences(sequences, max_length, padding_token=0):
    # Pack sequences
    packed_tokens = []
    cu_seqlens = [0]
    # ... implementation ...
    return packed_tokens, cu_seqlens
```

#### **Markdown documentation:**

Every significant feature should have:
1. **Overview**: What it does, why it's useful
2. **Quick start**: Minimal working example
3. **Configuration**: All options explained
4. **Examples**: Real-world usage
5. **Troubleshooting**: Common issues

✅ **DO:**
```markdown
# Speculative Decoding

## Overview

Speculative decoding speeds up inference by 2-3x using a small draft model
to propose tokens, which are then verified in parallel by the target model.

## Quick Start

\`\`\`bash
python train.py \
    --draft-model-path /path/to/small-model \
    --target-model-path /path/to/large-model \
    --spec-decode-lookahead 5 \
    # ... other args
\`\`\`

## Configuration

- `--draft-model-path`: Path to small draft model (e.g., 0.5B)
- `--spec-decode-lookahead`: Number of tokens to speculate (default: 5)
- `--spec-decode-threshold`: Acceptance threshold (default: 0.8)

## How It Works

1. Draft model generates K candidate tokens in parallel
2. Target model verifies all K tokens in one forward pass
3. Accept longest prefix that matches target model's distribution
4. Repeat from first rejected token

Expected speedup: 2-3x for K=5, draft model 10x smaller than target.

**Code reference:** `miles/rollout/sglang_rollout.py:177-182`

## Troubleshooting

### "Draft model too slow"
Use a smaller draft model (0.5B or less).

### "Low acceptance rate"
Reduce `--spec-decode-lookahead` or use a better draft model.
```

### 6.4 PR Process

#### **Creating a good PR:**

1. **Clear title:**
   - ✅ "Add support for Llama-4 architecture"
   - ✅ "Fix memory leak in gradient accumulation"
   - ✅ "Optimize data packing for 30% speedup"
   - ❌ "Fix bug"
   - ❌ "Update code"
   - ❌ "Changes"

2. **Descriptive PR description:**

```markdown
## Summary

Brief description of what this PR does.

## Motivation

Why is this change needed? What problem does it solve?

## Changes

- Added `MyNewFeature` class in `miles/utils/new_feature.py`
- Modified `miles/backends/fsdp_utils/actor.py` to use new feature
- Added tests in `tests/test_new_feature.py`
- Updated documentation in `docs/en/advanced/new-feature.md`

## Testing

How was this tested?

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Tested on 2-GPU setup
- [ ] Tested on 8-GPU setup

## Performance Impact

Benchmark results (if applicable):

- Before: 100 tokens/sec
- After: 130 tokens/sec (+30%)

## Breaking Changes

Does this PR break existing functionality? If yes, explain migration path.

## Checklist

- [ ] Code follows style guidelines (pre-commit passes)
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Added CI test label (run-ci-short or run-ci-long)
```

3. **Link related issues:**
   ```markdown
   Fixes #123
   Related to #456
   ```

4. **Request reviewers:**
   - Tag relevant maintainers
   - Explain any complex changes in PR comments

5. **Respond to feedback:**
   - Address all review comments
   - Mark conversations as resolved when fixed
   - Don't force-push after reviews (makes it hard to see changes)

#### **PR review checklist (for reviewers):**

- [ ] Code follows style guidelines
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No performance regressions
- [ ] No breaking changes (or properly documented)
- [ ] CI passes
- [ ] Changes are minimal and focused

---

## 7. Making Your First Contribution

### 7.1 Finding Good First Issues

**Look for issues labeled:**
- `good first issue`: Beginner-friendly
- `help wanted`: Maintainers need help
- `documentation`: Improve docs (great for learning the codebase!)
- `bug`: Fix a reported bug

**Ideas for first contributions:**

1. **Documentation improvements:**
   - Add missing docstrings
   - Fix typos in documentation
   - Add examples to existing docs
   - Create tutorial for a specific use case

2. **Small bug fixes:**
   - Fix edge cases in reward functions
   - Improve error messages
   - Handle missing arguments gracefully

3. **Test coverage:**
   - Add unit tests for untested functions
   - Add integration tests for features
   - Improve test documentation

4. **Code quality:**
   - Add type hints to untyped functions
   - Refactor complex functions for clarity
   - Remove dead code

### 7.2 Step-by-Step First Contribution

#### **Example: Adding a docstring to an undocumented function**

**Step 1: Find an undocumented function**

```bash
# Search for functions without docstrings
grep -r "def " miles/ --include="*.py" | grep -v '"""' | head -5
```

**Step 2: Fork and clone**

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/miles.git
cd miles
git remote add upstream https://github.com/radixark/miles.git
```

**Step 3: Create a branch**

```bash
git checkout -b docs/add-docstring-to-pack-sequences
```

**Step 4: Make your change**

```python
# Before
def pack_sequences(sequences, max_length):
    packed_tokens = []
    cu_seqlens = [0]
    # ... implementation ...
    return packed_tokens, cu_seqlens

# After
def pack_sequences(
    sequences: list[list[int]],
    max_length: int,
) -> tuple[list[int], list[int]]:
    """
    Pack variable-length sequences into a single sequence.

    Args:
        sequences: List of token sequences
        max_length: Maximum total length

    Returns:
        packed_tokens: Concatenated sequences
        cu_seqlens: Cumulative sequence lengths

    Example:
        >>> pack_sequences([[1, 2], [3, 4]], max_length=10)
        ([1, 2, 3, 4], [0, 2, 4])
    """
    packed_tokens = []
    cu_seqlens = [0]
    # ... implementation ...
    return packed_tokens, cu_seqlens
```

**Step 5: Test locally**

```bash
# Run pre-commit
pre-commit run --all-files

# Run relevant tests
pytest tests/ -k pack -v
```

**Step 6: Commit**

```bash
git add miles/backends/fsdp_utils/data_packing.py
git commit -m "docs: Add docstring to pack_sequences function

- Add comprehensive docstring with Args, Returns, Example
- Add type hints for clarity
- No functional changes"
```

**Step 7: Push and create PR**

```bash
git push origin docs/add-docstring-to-pack-sequences
```

Then create a PR on GitHub with a clear description.

**Step 8: Respond to reviews**

```bash
# Make requested changes
vim miles/backends/fsdp_utils/data_packing.py

# Commit changes
git add .
git commit -m "Address review feedback: improve example clarity"
git push origin docs/add-docstring-to-pack-sequences
```

### 7.3 More Advanced First Contributions

#### **Example: Adding a simple reward function**

**Goal:** Add a "length penalty" reward that penalizes overly long or short responses.

**Step 1: Implement the reward**

```python
# miles/rollout/rm_hub/length_penalty.py

from miles.utils.types import Sample

async def length_penalty_reward(args, sample: Sample) -> float:
    """
    Reward based on response length.

    Penalizes responses that are too short or too long relative to target.

    Args:
        args: Must have args.target_response_length (int)
        sample: Sample with response

    Returns:
        Reward between 0.0 and 1.0
    """
    response_length = len(sample.response.split())
    target_length = args.target_response_length

    # Compute distance from target
    diff = abs(response_length - target_length)

    # Exponential decay penalty
    penalty = diff / target_length
    reward = max(0.0, 1.0 - penalty)

    return reward
```

**Step 2: Register in hub**

```python
# miles/rollout/rm_hub/__init__.py

from .length_penalty import length_penalty_reward

async def async_rm(args, sample: Sample, **kwargs):
    # ... existing code ...

    elif rm_type == "length-penalty":
        return await length_penalty_reward(args, sample)

    # ... rest of function ...
```

**Step 3: Add argument**

```python
# miles/utils/arguments.py

def add_rollout_args(parser):
    # ... existing args ...

    group.add_argument(
        "--target-response-length",
        type=int,
        default=100,
        help="Target response length for length-penalty reward",
    )
```

**Step 4: Write tests**

```python
# tests/test_length_penalty.py

import pytest
from miles.utils.types import Sample
from miles.rollout.rm_hub.length_penalty import length_penalty_reward

class Args:
    target_response_length = 50

@pytest.mark.unit
async def test_length_penalty_exact_match():
    """Test reward for exact target length."""
    sample = Sample(response=" ".join(["word"] * 50))
    reward = await length_penalty_reward(Args(), sample)
    assert reward == 1.0

@pytest.mark.unit
async def test_length_penalty_too_short():
    """Test penalty for short response."""
    sample = Sample(response="short")  # 1 word vs target 50
    reward = await length_penalty_reward(Args(), sample)
    assert 0.0 < reward < 0.1  # High penalty

@pytest.mark.unit
async def test_length_penalty_too_long():
    """Test penalty for long response."""
    sample = Sample(response=" ".join(["word"] * 100))  # 100 words vs target 50
    reward = await length_penalty_reward(Args(), sample)
    assert 0.0 <= reward < 0.5  # Significant penalty
```

**Step 5: Run tests**

```bash
pytest tests/test_length_penalty.py -v
```

**Step 6: Add documentation**

```markdown
# docs/en/examples/length-penalty-reward.md

# Length Penalty Reward

## Overview

Rewards responses based on proximity to target length.

## Usage

\`\`\`bash
python train.py \
    --rm-type length-penalty \
    --target-response-length 100 \
    # ... other args
\`\`\`

## How It Works

Computes reward as: `max(0, 1 - |actual_length - target_length| / target_length)`

## Example

- Target: 100 words
- Response: 100 words → Reward: 1.0
- Response: 50 words → Reward: 0.5
- Response: 150 words → Reward: 0.5
```

**Step 7: Create PR**

```bash
git checkout -b feature/add-length-penalty-reward
git add miles/rollout/rm_hub/length_penalty.py \
        miles/rollout/rm_hub/__init__.py \
        miles/utils/arguments.py \
        tests/test_length_penalty.py \
        docs/en/examples/length-penalty-reward.md
git commit -m "feat: Add length penalty reward

- Add length_penalty_reward function
- Register in reward model hub
- Add --target-response-length argument
- Add comprehensive unit tests
- Add documentation with examples

Use case: Controlling response length during RL training"
git push origin feature/add-length-penalty-reward
```

**Step 8: Open PR with good description**

```markdown
## Summary

Add a length penalty reward function to encourage responses of target length.

## Motivation

Many tasks benefit from controlling response length (e.g., summarization).
This reward provides a simple way to penalize overly long or short responses.

## Changes

- New reward function: `miles/rollout/rm_hub/length_penalty.py`
- Registered in reward hub: `miles/rollout/rm_hub/__init__.py`
- New argument: `--target-response-length` in `miles/utils/arguments.py`
- Unit tests: `tests/test_length_penalty.py`
- Documentation: `docs/en/examples/length-penalty-reward.md`

## Testing

- [x] Unit tests pass (3/3)
- [x] Pre-commit hooks pass
- [ ] Tested in full training run (awaiting CI)

## Example Usage

\`\`\`bash
python train.py \
    --rm-type length-penalty \
    --target-response-length 100 \
    --prompt-data /data/summarization.parquet
\`\`\`
```

---

## Congratulations! 🎉

You've completed the Miles codebase learning journey!

### What You've Learned

**Steps 1-3 (Fundamentals):**
- ✅ Big picture: What Miles is and its three-tier architecture
- ✅ MoE architecture: Sparse experts, routing, memory challenges
- ✅ RL fundamentals: PPO, GAE, TD errors, policy collapse

**Steps 4-5 (Implementation):**
- ✅ Ray orchestration: Actor model, placement groups, distributed coordination
- ✅ Training loop: 8-step pipeline from rollout to gradient update
- ✅ Mathematical foundations: GAE backward iteration, PPO clipping, compound discounting
- ✅ Advanced features: Speculative decoding, memory management

**Step 6 (Practical):**
- ✅ Running training: Configuration scripts, command-line arguments
- ✅ Customization: Custom rewards, data sources, plugins
- ✅ Common patterns: SFT→RL, hyperparameter tuning, debugging

**Step 7 (Contributing):**
- ✅ Codebase navigation: Module organization, key files
- ✅ Development workflow: Pre-commit hooks, code style, testing
- ✅ CI/CD pipeline: GitHub Actions, test labels, GPU locking
- ✅ Contribution areas: Models, rewards, optimizations, docs
- ✅ Best practices: Code quality, testing, documentation, PR process

### Next Steps

**Continue learning:**
1. **Read the code:** Pick a feature you're interested in and trace through the implementation
2. **Run experiments:** Try different hyperparameters, reward functions, models
3. **Fix a bug:** Find an issue on GitHub and work on a fix
4. **Optimize:** Profile the code and improve performance
5. **Extend:** Add a feature you need for your research/project

**Get involved:**
1. ⭐ Star the repository: https://github.com/radixark/miles
2. 📖 Read the docs: https://github.com/radixark/miles/tree/main/docs
3. 💬 Join discussions: GitHub Discussions or issues
4. 🐛 Report bugs: Open an issue with reproduction steps
5. 🚀 Contribute: Follow this guide to make your first PR!

### Key Resources Reference

**Core files you'll frequently reference:**
- `miles/utils/ppo_utils.py:309-370` - GAE implementation
- `miles/utils/ppo_utils.py:125-148` - PPO loss
- `miles/backends/fsdp_utils/actor.py:531-766` - Training loop
- `miles/backends/fsdp_utils/data_packing.py` - Data packing
- `miles/rollout/sglang_rollout.py:90-199` - Generation
- `miles/ray/rollout.py` - Rollout orchestration
- `miles/utils/arguments.py` - All command-line arguments
- `miles/utils/types.py` - Core data structures

**Your learning materials:**
- `gae_explained.py` - GAE with concrete examples
- `ppo_explained.py` - PPO clipping explanation
- `why_gamma_lambda.py` - Compound discounting explained
- `data_packing_explained.md` - Visual guide to sequence packing
- `complete_training_loop_explained.md` - Full training pipeline
- `speculative_decoding_explained.md` - Draft-verify algorithm
- `STEP6_PRACTICAL_IMPLEMENTATION.md` - Practical guide
- `STEP7_CONTRIBUTING_TO_MILES.md` - This guide!

**Official documentation:**
- Quick start: `docs/en/get_started/quick_start.md`
- Examples: `docs/en/examples/`
- Advanced: `docs/en/advanced/`

### You're Ready to Contribute!

With your comprehensive understanding of:
- The mathematical foundations (PPO, GAE, RL concepts)
- The distributed architecture (Ray, FSDP, data packing)
- The training pipeline (rollout, advantages, loss, gradients)
- The codebase structure (modules, tests, CI/CD)
- The development workflow (pre-commit, testing, PRs)

**You can now:**
- Add new model architectures
- Implement custom reward functions
- Optimize training performance
- Fix bugs and improve code quality
- Write comprehensive tests and documentation
- Review PRs and help other contributors

**Welcome to the Miles community!** 🚀

---

*This guide was created as part of your comprehensive Miles learning journey. For questions or improvements to this guide, please open an issue or PR!*

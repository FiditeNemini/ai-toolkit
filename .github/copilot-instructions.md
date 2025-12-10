# AI Toolkit - Copilot Instructions

## ⚠️ macOS Port - Work in Progress

**Primary Goal**: Port this project to run on macOS with Apple Silicon (MPS backend). The codebase currently targets Windows/Linux with NVIDIA GPUs (CUDA). Key areas requiring adaptation:
- Replace CUDA-specific code with MPS-compatible alternatives
- Handle missing MPS operations (fallback to CPU where needed)
- Adapt quantization (bitsandbytes is CUDA-only; explore alternatives like `mlx` or CPU fallbacks)
- Test memory management on unified memory architecture

When making changes, prioritize MPS compatibility while maintaining existing CUDA functionality.

---

## 🔧 macOS/MPS Porting Guide

### High-Priority Files Requiring Changes

| File | Issue | Priority |
|------|-------|----------|
| `toolkit/optimizer.py` | `bitsandbytes` 8-bit optimizers are CUDA-only | **Critical** |
| `toolkit/cuda_malloc.py` | CUDA memory allocator setup - skip on MPS | **Critical** |
| `toolkit/memory_management/manager_modules.py` | CUDA streams/events for async transfers | **High** |
| `toolkit/util/quantize.py` | Uses `optimum.quanto` and `torchao` (check MPS support) | **High** |
| `toolkit/losses.py` | Hardcoded `torch.autocast(device_type='cuda')` | **Medium** |
| `toolkit/stable_diffusion_model.py` | Multiple `torch.cuda.empty_cache()` calls | **Medium** |
| `toolkit/models/base_model.py` | CUDA RNG state, empty_cache, manual_seed | **Medium** |
| `toolkit/models/flux.py` | Multi-GPU splitting with `cuda.device_count()` | **Medium** |
| `toolkit/train_tools.py` | `torch.cuda.manual_seed()` | **Medium** |
| `toolkit/sampler.py` | Hardcoded `pipe.to("cuda")` | **Medium** |
| `toolkit/control_generator.py` | Hardcoded `.to('cuda')` | **Medium** |
| `toolkit/optimizers/prodigy_8bit.py` | `.cuda()` call for distributed tensor | **Medium** |
| `jobs/process/models/critic.py` | `torch.cuda.amp.autocast(False)` | **Low** |
| `jobs/process/models/vgg19_critic.py` | `torch.cuda.amp.autocast(False)` | **Low** |
| `extensions_built_in/diffusion_models/omnigen2/src/ops/triton/` | Triton kernels (NVIDIA-only) | **Low** (optional model) |
| `extensions_built_in/diffusion_models/chroma/src/math.py` | flash_attn import, `.is_cuda` check | **Low** (optional model) |
| `extensions_built_in/diffusion_models/hidream/` | flash_attn dependency | **Low** (optional model) |
| `toolkit/models/pixtral_vision.py` | xformers memory_efficient_attention | **Low** |

### Files with Simple Device Selection Fixes

These files use `"cuda" if torch.cuda.is_available() else "cpu"` pattern - need MPS added:
- `toolkit/pixel_shuffle_encoder.py`
- `toolkit/layers.py`
- `toolkit/llvae.py`
- `toolkit/style.py`
- `toolkit/models/FakeVAE.py`
- `toolkit/models/diffusion_feature_extraction.py`
- `toolkit/models/wan21/wan21.py`
- `toolkit/models/wan21/wan21_i2v.py`
- `extensions_built_in/diffusion_models/wan22/wan22_pipeline.py`
- `extensions_built_in/dataset_tools/tools/fuyu_utils.py`
- `extensions_built_in/dataset_tools/tools/llava_utils.py`

### Files with `torch.cuda.empty_cache()` Calls

Need conditional MPS handling:
- `toolkit/basic.py`
- `toolkit/control_generator.py`
- `toolkit/stable_diffusion_model.py`
- `toolkit/models/base_model.py`
- `extensions_built_in/sd_trainer/SDTrainer.py`
- `extensions_built_in/advanced_generator/PureLoraGenerator.py`
- `extensions_built_in/advanced_generator/Img2ImgGenerator.py`
- `extensions_built_in/advanced_generator/ReferenceGenerator.py`
- `extensions_built_in/ultimate_slider_trainer/UltimateSliderTrainerProcess.py`
- `extensions_built_in/image_reference_slider_trainer/ImageReferenceSliderTrainerProcess.py`
- `extensions_built_in/dataset_tools/DatasetTools.py`
- `extensions_built_in/dataset_tools/SyncFromCollection.py`
- `extensions_built_in/dataset_tools/SuperTagger.py`
- `extensions_built_in/concept_replacer/ConceptReplacer.py`
- `extensions_built_in/diffusion_models/wan22/wan22_14b_model.py`

### Common Patterns to Replace

```python
# ❌ CUDA-specific
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.cuda.manual_seed(seed)
with torch.autocast(device_type='cuda'):
import bitsandbytes

# ✅ MPS-compatible
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# For empty_cache:
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()

# For autocast:
device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
with torch.autocast(device_type=device_type):
```

### Known MPS Limitations

1. **No Triton support** - Triton kernels only work on NVIDIA GPUs. The `omnigen2` extension uses Triton for layer norm - needs fallback to PyTorch ops
2. **Limited quantization** - `bitsandbytes` doesn't support MPS. Alternatives:
   - Use `optimum.quanto` (may have MPS support)
   - Use `torchao` with float8/uint configs
   - Fall back to fp16/bf16 without quantization
3. **No CUDA streams** - The `memory_management/manager_modules.py` ping-pong buffer system uses CUDA streams for async transfers. On MPS, use synchronous transfers or MPS-specific APIs
4. **GradScaler differences** - MPS autocast works differently; may need to disable GradScaler
5. **Some ops not implemented** - Check PyTorch MPS op coverage; use `.to('cpu')` fallback for unsupported ops
6. **flash_attn unavailable** - Flash Attention is CUDA-only. Used by `chroma`, `hidream`, `omnigen2` models. Use `F.scaled_dot_product_attention` fallback
7. **xformers limited** - xformers memory_efficient_attention may not work on MPS. Fallback to standard attention
8. **Multi-GPU code** - `toolkit/models/flux.py` uses `torch.cuda.device_count()` for multi-GPU splitting - needs MPS single-device path

### Optimizer Alternatives for MPS

In `toolkit/optimizer.py`, replace 8-bit optimizers:
```python
# Instead of bitsandbytes.optim.AdamW8bit:
# Option 1: Use standard AdamW
optimizer = torch.optim.AdamW(params, lr=learning_rate, eps=1e-6)

# Option 2: Use the toolkit's custom adam8bit (check MPS compatibility)
from toolkit.optimizers.adam8bit import Adam8bit
```

### Testing MPS Changes

1. Set environment: `export PYTORCH_ENABLE_MPS_FALLBACK=1` (allows CPU fallback for unsupported ops)
2. Test with small configs first (reduce resolution, batch_size=1)
3. Monitor memory with `torch.mps.current_allocated_memory()`

---

## Project Overview

AI Toolkit is an all-in-one training suite for diffusion models (FLUX.1, SDXL, SD3.5, Wan, etc.) supporting LoRA, LoKr, full fine-tuning, and slider training. It runs on consumer GPUs (24GB+ VRAM for FLUX.1) via CLI or web UI.

## Architecture

### Core Execution Flow
```
run.py → toolkit/job.py (get_job) → jobs/{ExtensionJob,TrainJob,...}.py → jobs/process/*.py
```

- **Entry point**: `run.py` - parses config files and dispatches to job handlers
- **Job system**: Jobs defined in `jobs/` load "processes" that execute the actual work
- **Extension system**: Most training uses `job: extension` with `type: sd_trainer` (see `extensions_built_in/sd_trainer/`)

### Key Architectural Patterns

1. **Config-driven execution**: All operations defined via YAML/JSON configs in `config/`. Use `[name]` tag for templating.
2. **Extension loading**: Extensions in `extensions/` and `extensions_built_in/` register via `AI_TOOLKIT_EXTENSIONS` list in `__init__.py`
3. **Process inheritance chain**: `BaseProcess` → `BaseTrainProcess` → `BaseSDTrainProcess` → `SDTrainer`

### Critical Directories
- `toolkit/` - Core library (model loading, training tools, data loading)
- `extensions_built_in/sd_trainer/` - Main LoRA/fine-tune trainer
- `jobs/process/` - Training process implementations
- `config/examples/` - Reference training configs

## Configuration System

Config files use this structure:
```yaml
job: extension  # or: train, extract, generate, mod
config:
  name: "my_lora"
  process:
    - type: 'sd_trainer'  # Extension uid from extensions_built_in/
      training_folder: "output"
      # ... training params
```

- Configs searched in: `config/` folder, then absolute path
- Environment vars via `${VAR_NAME}` syntax (see `toolkit/config.py`)
- `[name]` replaced with `config.name` value

## Training Workflow

### Dataset Format
- Folder with images (`.jpg`, `.jpeg`, `.png`) + matching `.txt` caption files
- Images auto-resized/bucketed; no manual cropping needed
- Use `[trigger]` in captions, replaced by `trigger_word` config

### Key Config Sections
```yaml
network:
  type: "lora"  # or: lokr, locon, lorm
  linear: 16
  linear_alpha: 16
  network_kwargs:
    only_if_contains: ["transformer.single_transformer_blocks."]  # Train specific layers
    ignore_if_contains: []  # Exclude layers

model:
  name_or_path: "black-forest-labs/FLUX.1-dev"
  is_flux: true
  quantize: true  # 8-bit mixed precision for 24GB GPUs
  low_vram: true  # For GPU with monitors attached

train:
  dtype: bf16  # Required for FLUX
  gradient_checkpointing: true
  noise_scheduler: "flowmatch"  # For FLUX/flow-based models
```

### FLUX.1 Specifics
- Requires HF token in `.env` file: `HF_TOKEN=your_key`
- For schnell: add `assistant_lora_path: "ostris/FLUX.1-schnell-training-adapter"`
- Minimum 24GB VRAM; use `low_vram: true` if GPU has monitors

## Code Patterns

### Adding a New Extension
1. Create folder in `extensions/your_extension/`
2. Define extension class inheriting `Extension` (see `toolkit/extension.py`)
3. Register in `__init__.py` via `AI_TOOLKIT_EXTENSIONS = [YourExtension]`
4. Implement `get_process()` returning your process class

### Config Classes
All config objects in `toolkit/config_modules.py` use kwargs pattern:
```python
class TrainConfig:
    def __init__(self, **kwargs):
        self.batch_size: int = kwargs.get('batch_size', 1)
```

### Model Abstraction
`toolkit/stable_diffusion_model.py` provides `StableDiffusion` class wrapping all diffusion models with unified interface for:
- Loading (diffusers/safetensors)
- Quantization
- Text encoding
- Sampling

## Development Commands

```bash
# Training via CLI
python run.py config/your_config.yaml

# Training via UI (Next.js + Python backend)
cd ui && npm run build_and_start

# With auth token for exposed servers
AI_TOOLKIT_AUTH=password npm run build_and_start

# Resume training
# Just re-run same command - picks up from last checkpoint

# Multi-config sequential run
python run.py config1.yaml config2.yaml --recover
```

## Environment Setup

```bash
# Required: Python 3.10+, CUDA GPU
python3 -m venv venv && source venv/bin/activate
pip3 install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
```

Critical env vars:
- `HF_TOKEN` - Hugging Face token for gated models
- `AI_TOOLKIT_AUTH` - UI authentication
- `DEBUG_TOOLKIT=1` - Enable torch anomaly detection
- `MODELS_PATH` - Custom models directory

## Common Pitfalls

- **Ctrl+C during save corrupts checkpoints** - Wait for save completion
- **Webp images have issues** - Use jpg/jpeg/png for datasets
- **Windows native may have bugs** - WSL recommended
- **Text encoder training with FLUX** - Generally doesn't work well (`train_text_encoder: false`)

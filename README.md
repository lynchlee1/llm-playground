# LLM Playground — Qwen3.5-9B MLX Workspace

Local inference workspace for **Qwen3.5-9B** quantised to **5-bit MLX** (`Q5`).  
Processes a `list[str]` one item at a time with fully configurable sampling parameters.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Apple Silicon Mac (M1/M2/M3/M4) | MLX only runs on Apple Silicon |
| macOS 14 Sonoma or later | Recommended |
| Python ≥ 3.10 | `brew install python@3.12` |
| ~16 GB disk space | Original weights (~10 GB) + Q5 model (~6 GB) |

---

## 1. Install dependencies

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package and its runtime dependencies (mlx, mlx-lm)
pip install -e .

# Optional: install dev/test extras
pip install -e ".[dev]"
```

---

## 2. Download & quantise the model

Run the provided conversion script.  
It will download the original Qwen3.5-9B weights from Hugging Face and convert them to a 5-bit MLX model.

```bash
bash scripts/convert_model.sh
```

This is equivalent to running:

```bash
# Download
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3.5-9B', local_dir='./Qwen3.5-9B', local_dir_use_symlinks=False)
"

# Convert
python -m mlx_lm.convert \
    --model ./Qwen3.5-9B \
    --quantize \
    --q-bits 5 \
    --output-path ./Qwen3.5-9B-MLX-Q5
```

The quantised model lands in `./Qwen3.5-9B-MLX-Q5/` (excluded from git).

---

## 3. Usage

### Quick example

```python
from qwen_parser import QwenParser, InferenceConfig

# Tune sampling parameters here
cfg = InferenceConfig(
    model_path="./Qwen3.5-9B-MLX-Q5",
    temperature=0.3,   # lower = more deterministic
    top_p=0.85,        # nucleus sampling threshold
    top_k=0,           # 0 = disabled
    max_tokens=512,
    repetition_penalty=1.1,
    system_prompt="Parse the text and extract key fields as JSON.",
)

parser = QwenParser(cfg)  # model loads once here

items: list[str] = [
    "Order #12345 placed on 2024-01-15 for $99.99",
    "Flight AA100 departs JFK 08:30, arrives LAX 11:45",
    "Patient: John Doe | DOB: 1985-03-22 | BP: 120/80",
]

results: list[str] = parser.parse_batch(items)

for original, parsed in zip(items, results):
    print(f"INPUT : {original}")
    print(f"OUTPUT: {parsed}\n")
```

### Streaming a single item

```python
for chunk in parser.stream_parse(items[0]):
    print(chunk, end="", flush=True)
```

---

## 4. Configuration reference

All parameters are set via `InferenceConfig`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | `./Qwen3.5-9B-MLX-Q5` | Path to quantised model directory |
| `temperature` | `float` | `0.7` | Sampling temperature (0 = greedy) |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold |
| `top_k` | `int` | `0` | Top-k sampling (0 = disabled) |
| `max_tokens` | `int` | `512` | Maximum tokens to generate |
| `repetition_penalty` | `float` | `1.0` | Repetition penalty (1.0 = off) |
| `repetition_context_size` | `int` | `20` | Window size for repetition penalty |
| `system_prompt` | `str` | *(see config.py)* | System message prepended to every call |

---

## 5. Project layout

```
.
├── pyproject.toml              # project metadata and dependencies
├── scripts/
│   └── convert_model.sh        # download + quantise script
└── src/
    └── qwen_parser/
        ├── __init__.py
        ├── config.py           # InferenceConfig dataclass
        └── parser.py           # QwenParser class
```

Model directories (`Qwen3.5-9B/`, `Qwen3.5-9B-MLX-Q5/`) are git-ignored.

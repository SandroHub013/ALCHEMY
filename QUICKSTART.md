# ⚡ Quick Start Guide

Get started with LLM Fine-tuning with Agent Lightning in 5 minutes.

---

## 📋 Prerequisites

- **Python 3.10+**
- **CUDA GPU** (8GB+ VRAM recommended — tested on RTX 2070 Super 8GB)
- **Git**

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SandroHub013/ALCHEMY.git
cd ALCHEMY
```

### 2. Create a Virtual Environment

```bash
# With venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Or with conda
conda create -n llm-finetune python=3.10
conda activate llm-finetune
```

### 3. Install Dependencies

```bash
# Full installation (includes Agent Lightning)
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# NOTE WINDOWS: if bitsandbytes gives an error
pip uninstall bitsandbytes
pip install bitsandbytes-windows
```

### 4. Verify Installation

```bash
python scripts/check_installation.py
```

### 5. Verify CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 🎯 Your First Training

### Option 1: Classic Training (SFT)

```bash
python main.py --config config/config.yaml
```

### Option 2: Training with Reinforcement Learning

```bash
python main_agent_lightning.py --config config/config.yaml
```

### Option 3: Dry Run (Verify Configuration)

```bash
python main_agent_lightning.py --config config/config.yaml --dry-run
```

---

## 📝 Quick Configuration

### Minimal Configuration

Create `config/my_config.yaml`:

```yaml
model:
  name_or_path: "mistralai/Mistral-7B-v0.3"

peft:
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
  lora:
    r: 16
    lora_alpha: 32

training:
  num_epochs: 1
  per_device_train_batch_size: 1
  learning_rate: 2.0e-4
  output_dir: "./checkpoints"

datasets:
  multi_source_enabled: false

data:
  dataset_name: "databricks/databricks-dolly-15k"

agent_lightning:
  enabled: false
```

### Run with Your Config

```bash
python main.py --config config/my_config.yaml
```

---

## 🧠 Using the RAG System

### Document Ingestion

```python
from src.memory import VectorStore, create_vector_store

# Create vector store
store = create_vector_store(persist_path="./my_kb")

# Add documents
store.add_documents([
    "Python is a versatile programming language.",
    "PyTorch is a deep learning framework.",
    "Transformers are the foundation of modern LLMs.",
])

print(f"Documents in database: {store.count()}")
```

### Query

```python
# Search
results = store.query("What is Python?", n_results=3)

for doc, score, meta in results:
    print(f"[{score:.3f}] {doc[:100]}...")
```

---

## 📋 Using SOPs

```python
from src.memory import SOPManager, get_system_prompt_with_sop

# Load SOPs
manager = SOPManager(sop_directory="./data/sops")

# Generate prompt with appropriate procedure
query = "Help me debug this code"
system_prompt = get_system_prompt_with_sop(query, manager)

print(system_prompt)
```

---

## 🔍 Using Smart Chunker

```python
from src.memory import SmartChunker

chunker = SmartChunker(
    max_chunk_size=2000,
    min_chunk_size=100,
)

# Chunk a file
chunks = chunker.chunk_file("src/memory/vector_store.py")

for chunk in chunks:
    print(f"[{chunk.chunk_type.value}] {chunk.qualified_name}: {chunk.char_count} chars")
```

---

## 📊 Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/llm_finetuning
```

Then open http://localhost:6006

---

## ❓ Troubleshooting

### "CUDA out of memory"

```yaml
# Reduce batch size
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

### "bitsandbytes not found"

```bash
# Linux
pip install bitsandbytes

# Windows (may require building from source)
pip install bitsandbytes-windows
```

### "agentlightning not found"

```bash
pip install agentlightning

# Or use classic training
python main.py --config config/config.yaml
```

---

## 📚 Next Steps

1. 📖 Read [README.md](README.md) for a complete overview
2. 🏗️ Explore [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system
3. ✨ Consult [FEATURES.md](FEATURES.md) for all features
4. 🎬 See [SHOWCASE.md](SHOWCASE.md) for real examples

---

*Having issues? Open an [Issue](https://github.com/SandroHub013/ALCHEMY/issues)!*

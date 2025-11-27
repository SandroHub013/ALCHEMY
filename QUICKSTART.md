# ⚡ Quick Start Guide

Inizia a usare LLM Fine-tuning con Agent Lightning in 5 minuti.

---

## 📋 Prerequisiti

- **Python 3.10+**
- **CUDA GPU** (consigliato: 16GB+ VRAM per modelli 7B)
- **Git**

---

## 🚀 Installazione

### 1. Clona il Repository

```bash
git clone https://github.com/tuousername/llm-finetuning-agent-lightning.git
cd llm-finetuning-agent-lightning
```

### 2. Crea un Ambiente Virtuale

```bash
# Con venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Oppure con conda
conda create -n llm-finetune python=3.10
conda activate llm-finetune
```

### 3. Installa le Dipendenze

```bash
# Installazione completa (include Agent Lightning)
pip install -e .

# Con dipendenze di sviluppo
pip install -e ".[dev]"

# NOTA WINDOWS: se bitsandbytes dà errore
pip uninstall bitsandbytes
pip install bitsandbytes-windows
```

### 4. Verifica Installazione

```bash
python scripts/check_installation.py
```

### 5. Verifica CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 🎯 Il Tuo Primo Training

### Opzione 1: Training Classico (SFT)

```bash
python main.py --config config/config.yaml
```

### Opzione 2: Training con Reinforcement Learning

```bash
python main_agent_lightning.py --config config/config.yaml
```

### Opzione 3: Dry Run (Verifica Configurazione)

```bash
python main_agent_lightning.py --config config/config.yaml --dry-run
```

---

## 📝 Configurazione Rapida

### Configurazione Minima

Crea `config/my_config.yaml`:

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

### Esegui con la Tua Config

```bash
python main.py --config config/my_config.yaml
```

---

## 🧠 Usa il Sistema RAG

### Ingestione Documenti

```python
from src.memory import VectorStore, create_vector_store

# Crea vector store
store = create_vector_store(persist_path="./my_kb")

# Aggiungi documenti
store.add_documents([
    "Python è un linguaggio di programmazione versatile.",
    "PyTorch è un framework per deep learning.",
    "I Transformer sono alla base dei moderni LLM.",
])

print(f"Documenti nel database: {store.count()}")
```

### Query

```python
# Cerca
results = store.query("Cos'è Python?", n_results=3)

for doc, score, meta in results:
    print(f"[{score:.3f}] {doc[:100]}...")
```

---

## 📋 Usa le SOP

```python
from src.memory import SOPManager, get_system_prompt_with_sop

# Carica SOP
manager = SOPManager(sop_directory="./data/sops")

# Genera prompt con procedura appropriata
query = "Aiutami a debuggare questo codice"
system_prompt = get_system_prompt_with_sop(query, manager)

print(system_prompt)
```

---

## 🔍 Usa lo Smart Chunker

```python
from src.memory import SmartChunker

chunker = SmartChunker(
    max_chunk_size=2000,
    min_chunk_size=100,
)

# Chunka un file
chunks = chunker.chunk_file("src/memory/vector_store.py")

for chunk in chunks:
    print(f"[{chunk.chunk_type.value}] {chunk.qualified_name}: {chunk.char_count} chars")
```

---

## 📊 Monitoraggio

### TensorBoard

```bash
tensorboard --logdir logs/llm_finetuning
```

Poi apri http://localhost:6006

---

## ❓ Troubleshooting

### "CUDA out of memory"

```yaml
# Riduci batch size
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

### "bitsandbytes not found"

```bash
# Linux
pip install bitsandbytes

# Windows (potrebbe richiedere build da source)
pip install bitsandbytes-windows
```

### "agentlightning not found"

```bash
pip install agentlightning

# Oppure usa training classico
python main.py --config config/config.yaml
```

---

## 📚 Prossimi Passi

1. 📖 Leggi [README.md](README.md) per una panoramica completa
2. 🏗️ Esplora [ARCHITECTURE.md](ARCHITECTURE.md) per capire il sistema
3. ✨ Consulta [FEATURES.md](FEATURES.md) per tutte le funzionalità
4. 🎬 Vedi [SHOWCASE.md](SHOWCASE.md) per esempi reali

---

*Hai problemi? Apri una [Issue](https://github.com/tuousername/llm-finetuning-agent-lightning/issues)!*


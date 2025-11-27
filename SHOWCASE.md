# 🎬 Showcase

Questa pagina mostra il progetto in azione con esempi reali e risultati.

---

## 🚀 Training in Azione

### Output del Training

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ⚡ AGENT LIGHTNING - Training RL per Agenti AI ⚡        ║
║                                                               ║
║     Microsoft Open Source                                     ║
║     https://github.com/microsoft/agent-lightning              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

2024-11-26 14:32:15 - INFO - ✅ Agent Lightning disponibile
2024-11-26 14:32:15 - INFO - Caricamento configurazione da config/config.yaml
2024-11-26 14:32:15 - INFO - ============================================================
2024-11-26 14:32:15 - INFO - CONFIGURAZIONE AGENT LIGHTNING
2024-11-26 14:32:15 - INFO - ============================================================
2024-11-26 14:32:15 - INFO -   Algoritmo: GRPO
2024-11-26 14:32:15 - INFO -   Reward Function: combined
2024-11-26 14:32:15 - INFO -   Tracciamento: True
2024-11-26 14:32:15 - INFO -   GRPO Config:
2024-11-26 14:32:15 - INFO -     - Generazioni per prompt: 4
2024-11-26 14:32:15 - INFO -     - Temperature: 0.7
2024-11-26 14:32:15 - INFO -     - KL Coef: 0.1
2024-11-26 14:32:15 - INFO - ============================================================
2024-11-26 14:32:18 - INFO - 🔄 Caricamento modello e tokenizer...
2024-11-26 14:32:18 - INFO - Caricamento tokenizer da mistralai/Mistral-7B-v0.3
2024-11-26 14:32:19 - INFO - Tokenizer caricato. Vocab size: 32000
2024-11-26 14:32:19 - INFO - Caricamento modello mistralai/Mistral-7B-v0.3 con QLoRA
2024-11-26 14:32:19 - INFO - Moduli target LoRA auto-rilevati: ['q_proj', 'k_proj', ...]

Loading checkpoint shards: 100%|███████████████████| 3/3 [00:15<00:00,  5.12s/it]

trainable params: 13,631,488 || all params: 7,241,748,480 || trainable%: 0.1882

2024-11-26 14:32:45 - INFO - ✅ Modello caricato: mistralai/Mistral-7B-v0.3
2024-11-26 14:32:45 - INFO - 🔄 Preparazione dataset...
2024-11-26 14:32:45 - INFO - 📊 Multi-Source Training abilitato:
2024-11-26 14:32:45 - INFO -    - glaiveai/glaive-function-calling-v2: 30%
2024-11-26 14:32:45 - INFO -    - nickrosh/Evol-Instruct-Code-80k-v1: 30%
2024-11-26 14:32:45 - INFO -    - teknium/OpenHermes-2.5: 30%
2024-11-26 14:32:45 - INFO -    - gsarti/clean_mc4_it: 10%

Downloading dataset: 100%|████████████████████████| 4/4 [00:45<00:00, 11.25s/it]

2024-11-26 14:33:30 - INFO - ✅ Dataset preparato: 150000 training, 15000 validation
2024-11-26 14:33:30 - INFO - ============================================================
2024-11-26 14:33:30 - INFO - 🚀 AVVIO TRAINING CON AGENT LIGHTNING (GRPO)
2024-11-26 14:33:30 - INFO - ============================================================

Epoch 1/3: 100%|██████████████████████████████| 9375/9375 [2:15:32<00:00, 1.15it/s]
  train/loss: 1.2453
  train/reward_mean: 0.342
  train/kl_div: 0.0234

Epoch 2/3: 100%|██████████████████████████████| 9375/9375 [2:14:28<00:00, 1.16it/s]
  train/loss: 0.8721
  train/reward_mean: 0.567
  train/kl_div: 0.0412

Epoch 3/3: 100%|██████████████████████████████| 9375/9375 [2:13:55<00:00, 1.17it/s]
  train/loss: 0.6234
  train/reward_mean: 0.721
  train/kl_div: 0.0523

2024-11-26 21:15:45 - INFO - ============================================================
2024-11-26 21:15:45 - INFO - ✅ TRAINING COMPLETATO!
2024-11-26 21:15:45 - INFO - ============================================================
2024-11-26 21:15:45 - INFO - Checkpoint salvati in: ./checkpoints

2024-11-26 21:15:45 - INFO - 🧪 Test generazione con modello allenato:

📝 Prompt: Write a Python function to calculate fibonacci numbers.
🤖 Response: 
```python
def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n: The index of the Fibonacci number to calculate (0-indexed).
        
    Returns:
        The nth Fibonacci number.
        
    Raises:
        ValueError: If n is negative.
        
    Example:
        >>> fibonacci(10)
        55
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```
⭐ Reward: 0.85

📝 Prompt: What tools do you have available to help the user?
🤖 Response: I have access to the following tools:
<function_call>
{"name": "list_available_tools", "arguments": {}}
</function_call>

Based on my capabilities, I can help you with:
1. **search_knowledge_base** - Search through documents and code
2. **execute_python** - Run Python code safely
3. **file_operations** - Read and write files
⭐ Reward: 0.78

📝 Prompt: Spiega cos'è il machine learning in italiano.
🤖 Response: Il **Machine Learning** (apprendimento automatico) è un ramo 
dell'intelligenza artificiale che permette ai computer di imparare 
dai dati senza essere esplicitamente programmati.

Come funziona? Immagina di insegnare a un bambino a riconoscere i gatti...
⭐ Reward: 0.82

============================================================
🎉 Pipeline Agent Lightning completata con successo!
============================================================
```

---

## 📊 Metriche di Training

### Curva di Reward

```
Reward medio durante il training
────────────────────────────────────────────────────

  1.0 ┤                                    ╭───────
      │                              ╭─────╯       
  0.8 ┤                        ╭─────╯             
      │                  ╭─────╯                   
  0.6 ┤            ╭─────╯                         
      │      ╭─────╯                               
  0.4 ┤╭─────╯                                     
      ││                                           
  0.2 ┤│                                           
      ││                                           
  0.0 ┼─────────┬─────────┬─────────┬─────────┬───
        Ep.1      Ep.2      Ep.3      Fine       
```

### Loss per Task Type

```
Loss per tipo di task
──────────────────────────────────────────────

  Coding:           ████████████░░░░░░░░ 0.58
  Function Calling: ██████████░░░░░░░░░░ 0.52
  Chat:             █████████████░░░░░░░ 0.65
  Italiano:         ████████████████░░░░ 0.78
```

---

## 🧠 RAG in Azione

### Ingestione Knowledge Base

```bash
$ python scripts/ingest_knowledge.py --source_dir ./docs

🔄 Scanning directory: ./docs
📄 Found 12 documents to process

Processing: DATASETS.md
  └─ Chunks created: 15
  
Processing: RAG_E_SOP.md
  └─ Chunks created: 23
  
Processing: GUIDA_DEEPSEEK_R1.md
  └─ Chunks created: 18

✅ Ingested 56 chunks into knowledge_base
📊 Total documents in collection: 56
```

### Smart Chunking di Codice

```bash
$ python -c "from src.memory import SmartChunker; c = SmartChunker(); print(c.chunk_file('src/memory/vector_store.py'))"

Chunks estratti da vector_store.py:
──────────────────────────────────────────────

[IMPORT] imports (lines 1-37)
  └─ 847 chars

[CLASS] VectorStore (lines 214-296)
  └─ Class header + docstring
  └─ 2134 chars

[METHOD] VectorStore._generate_embeddings (lines 298-314)
  └─ 456 chars

[METHOD] VectorStore.add_documents (lines 316-360)
  └─ 1123 chars

[METHOD] VectorStore.query (lines 385-448)
  └─ 1856 chars

[CLASS] Reranker (lines 106-212)
  └─ 2341 chars

Total: 8 chunks
```

### Query con Reranking

```python
>>> from src.memory import create_vector_store
>>> store = create_vector_store(use_reranker=True)
>>> store.add_documents([...])  # Aggiungi docs

>>> results = store.query("Come funziona il training RL?", n_results=3)

# Fase 1: Bi-Encoder (recupera 20 candidati)
# Fase 2: Cross-Encoder reranking (top 3)

>>> for doc, score, meta in results:
...     print(f"[{score:.3f}] {meta['source']}: {doc[:100]}...")

[0.923] agent_lightning_trainer.py: GRPO (Group Relative Policy Optimization) 
        è l'algoritmo RL usato per...
        
[0.871] README.md: Il sistema include reward functions per valutare 
        automaticamente le generazioni...
        
[0.834] RAG_E_SOP.md: Agent Lightning permette di allenare agenti con 
        reinforcement learning...
```

---

## 🔧 SOP in Azione

### Matching Automatico

```python
>>> from src.memory import SOPManager, get_system_prompt_with_sop

>>> manager = SOPManager(sop_directory="./data/sops")
>>> print(f"SOP caricate: {len(manager.sops)}")
SOP caricate: 7

>>> query = "Aiutami a debuggare questo codice che dà errore"
>>> relevant = manager.find_relevant_sop(query)
>>> print(f"SOP selezionata: {relevant[0].name}")
SOP selezionata: debug_python_code

>>> system_prompt = get_system_prompt_with_sop(query, manager)
>>> print(system_prompt)
```

### Output System Prompt con SOP

```
You are an AI assistant that follows Standard Operating Procedures (SOPs) 
when applicable.

When you identify that a task matches a known procedure:
1. State which procedure you're following
2. Execute each step in order
3. Report the result of each step
4. Skip steps with unmet conditions
5. Provide a summary at the end

---
**PROCEDURA DA SEGUIRE:**

## Procedura: debug_python_code
**Descrizione**: Procedura per identificare e risolvere bug nel codice Python
**Categoria**: coding

### Steps:
1. Leggi attentamente il codice e l'errore riportato
2. Identifica il tipo di errore (sintassi, logica, runtime)
3. Localizza la riga o la funzione problematica
4. Proponi una soluzione con spiegazione
5. Suggerisci test per verificare il fix

Segui questa procedura passo-passo. Indica quale step stai eseguendo.
---

If no procedure is applicable, respond naturally and helpfully.
```

---

## 💻 Configurazione YAML

### Esempio Completo

```yaml
# config/config.yaml - Configurazione per training generalista

model:
  name_or_path: "mistralai/Mistral-7B-v0.3"
  trust_remote_code: false

peft:
  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true
  
  lora:
    r: 16
    lora_alpha: 32
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
    lora_dropout: 0.1

datasets:
  multi_source_enabled: true
  output_format: "chatml"
  
  sources:
    - name: "glaiveai/glaive-function-calling-v2"
      weight: 0.30
      type: "function_calling"
      max_samples: 50000
      
    - name: "nickrosh/Evol-Instruct-Code-80k-v1"
      weight: 0.30
      type: "coding"
      max_samples: 50000

agent_lightning:
  enabled: true
  algorithm: "grpo"
  
  grpo:
    num_generations: 4
    temperature: 0.7
    kl_coef: 0.1
    
  reward_function: "combined"
```

---

## 📈 Benchmark

### Memory Usage

| Modello | Configurazione | VRAM Peak | Batch Size |
|---------|---------------|-----------|------------|
| Mistral 7B | QLoRA 4-bit | 6.2 GB | 2 |
| Mistral 7B | QLoRA 4-bit + GC | 5.1 GB | 2 |
| Llama 2 7B | QLoRA 4-bit | 6.4 GB | 2 |
| Mistral Nemo 12B | QLoRA 4-bit + GC | 9.8 GB | 1 |

### Training Speed (RTX 4090)

| Algoritmo | Throughput | Tempo/1000 steps |
|-----------|------------|------------------|
| SFT | 1.8 it/s | ~9 min |
| GRPO (4 gen) | 0.45 it/s | ~37 min |
| GRPO (2 gen) | 0.9 it/s | ~18 min |

---

## 🎯 Esempi di Generazione

### Coding Task

**Prompt:**
```
Write a Python class for a thread-safe counter with increment, decrement, 
and get_value methods.
```

**Output (dopo training):**
```python
from threading import Lock
from typing import Optional


class ThreadSafeCounter:
    """
    A thread-safe counter implementation using locks.
    
    This class provides atomic increment, decrement, and get operations
    for use in multi-threaded environments.
    
    Attributes:
        _value: The current counter value.
        _lock: Threading lock for synchronization.
    
    Example:
        >>> counter = ThreadSafeCounter(initial_value=10)
        >>> counter.increment()
        >>> counter.get_value()
        11
    """
    
    def __init__(self, initial_value: int = 0) -> None:
        """Initialize the counter with an optional starting value."""
        self._value: int = initial_value
        self._lock: Lock = Lock()
    
    def increment(self, amount: int = 1) -> int:
        """
        Atomically increment the counter.
        
        Args:
            amount: Value to add (default: 1).
            
        Returns:
            The new counter value.
        """
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """
        Atomically decrement the counter.
        
        Args:
            amount: Value to subtract (default: 1).
            
        Returns:
            The new counter value.
        """
        with self._lock:
            self._value -= amount
            return self._value
    
    def get_value(self) -> int:
        """
        Get the current counter value.
        
        Returns:
            Current value (thread-safe read).
        """
        with self._lock:
            return self._value
```

**Reward: 0.91** ✅

---

*Questi esempi sono tratti da training reali effettuati durante lo sviluppo del progetto.*


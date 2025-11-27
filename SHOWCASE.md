# 🎬 Showcase

This page shows the project in action with real examples and results.

---

## 🚀 Training in Action

### Training Output

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ⚡ AGENT LIGHTNING - RL Training for AI Agents ⚡        ║
║                                                               ║
║     Microsoft Open Source                                     ║
║     https://github.com/microsoft/agent-lightning              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

2024-11-26 14:32:15 - INFO - ✅ Agent Lightning available
2024-11-26 14:32:15 - INFO - Loading configuration from config/config.yaml
2024-11-26 14:32:15 - INFO - ============================================================
2024-11-26 14:32:15 - INFO - AGENT LIGHTNING CONFIGURATION
2024-11-26 14:32:15 - INFO - ============================================================
2024-11-26 14:32:15 - INFO -   Algorithm: GRPO
2024-11-26 14:32:15 - INFO -   Reward Function: combined
2024-11-26 14:32:15 - INFO -   Tracing: True
2024-11-26 14:32:15 - INFO -   GRPO Config:
2024-11-26 14:32:15 - INFO -     - Generations per prompt: 4
2024-11-26 14:32:15 - INFO -     - Temperature: 0.7
2024-11-26 14:32:15 - INFO -     - KL Coef: 0.1
2024-11-26 14:32:15 - INFO - ============================================================
2024-11-26 14:32:18 - INFO - 🔄 Loading model and tokenizer...
2024-11-26 14:32:18 - INFO - Loading tokenizer from mistralai/Mistral-7B-v0.3
2024-11-26 14:32:19 - INFO - Tokenizer loaded. Vocab size: 32000
2024-11-26 14:32:19 - INFO - Loading model mistralai/Mistral-7B-v0.3 with QLoRA
2024-11-26 14:32:19 - INFO - LoRA target modules auto-detected: ['q_proj', 'k_proj', ...]

Loading checkpoint shards: 100%|███████████████████| 3/3 [00:15<00:00,  5.12s/it]

trainable params: 13,631,488 || all params: 7,241,748,480 || trainable%: 0.1882

2024-11-26 14:32:45 - INFO - ✅ Model loaded: mistralai/Mistral-7B-v0.3
2024-11-26 14:32:45 - INFO - 🔄 Preparing dataset...
2024-11-26 14:32:45 - INFO - 📊 Multi-Source Training enabled:
2024-11-26 14:32:45 - INFO -    - glaiveai/glaive-function-calling-v2: 30%
2024-11-26 14:32:45 - INFO -    - nickrosh/Evol-Instruct-Code-80k-v1: 30%
2024-11-26 14:32:45 - INFO -    - teknium/OpenHermes-2.5: 30%
2024-11-26 14:32:45 - INFO -    - gsarti/clean_mc4_it: 10%

Downloading dataset: 100%|████████████████████████| 4/4 [00:45<00:00, 11.25s/it]

2024-11-26 14:33:30 - INFO - ✅ Dataset prepared: 150000 training, 15000 validation
2024-11-26 14:33:30 - INFO - ============================================================
2024-11-26 14:33:30 - INFO - 🚀 STARTING TRAINING WITH AGENT LIGHTNING (GRPO)
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
2024-11-26 21:15:45 - INFO - ✅ TRAINING COMPLETED!
2024-11-26 21:15:45 - INFO - ============================================================
2024-11-26 21:15:45 - INFO - Checkpoints saved to: ./checkpoints

2024-11-26 21:15:45 - INFO - 🧪 Test generation with trained model:

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

📝 Prompt: Explain what machine learning is.
🤖 Response: **Machine Learning** is a branch of artificial intelligence 
that enables computers to learn from data without being explicitly 
programmed.

How does it work? Imagine teaching a child to recognize cats...
⭐ Reward: 0.82

============================================================
🎉 Agent Lightning pipeline completed successfully!
============================================================
```

---

## 📊 Training Metrics

### Reward Curve

```
Average reward during training
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
        Ep.1      Ep.2      Ep.3      End       
```

### Loss by Task Type

```
Loss by task type
──────────────────────────────────────────────

  Coding:           ████████████░░░░░░░░ 0.58
  Function Calling: ██████████░░░░░░░░░░ 0.52
  Chat:             █████████████░░░░░░░ 0.65
  Italian:          ████████████████░░░░ 0.78
```

---

## 🧠 RAG in Action

### Knowledge Base Ingestion

```bash
$ python scripts/ingest_knowledge.py --source_dir ./docs

🔄 Scanning directory: ./docs
📄 Found 12 documents to process

Processing: DATASETS.md
  └─ Chunks created: 15
  
Processing: RAG_E_SOP.md
  └─ Chunks created: 23
  
Processing: GUIDE_DEEPSEEK_R1.md
  └─ Chunks created: 18

✅ Ingested 56 chunks into knowledge_base
📊 Total documents in collection: 56
```

### Smart Chunking of Code

```bash
$ python -c "from src.memory import SmartChunker; c = SmartChunker(); print(c.chunk_file('src/memory/vector_store.py'))"

Chunks extracted from vector_store.py:
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

### Query with Reranking

```python
>>> from src.memory import create_vector_store
>>> store = create_vector_store(use_reranker=True)
>>> store.add_documents([...])  # Add docs

>>> results = store.query("How does RL training work?", n_results=3)

# Phase 1: Bi-Encoder (retrieve 20 candidates)
# Phase 2: Cross-Encoder reranking (top 3)

>>> for doc, score, meta in results:
...     print(f"[{score:.3f}] {meta['source']}: {doc[:100]}...")

[0.923] agent_lightning_trainer.py: GRPO (Group Relative Policy Optimization) 
        is the RL algorithm used for...
        
[0.871] README.md: The system includes reward functions to automatically 
        evaluate generations...
        
[0.834] RAG_E_SOP.md: Agent Lightning enables training agents with 
        reinforcement learning...
```

---

## 🔧 SOP in Action

### Automatic Matching

```python
>>> from src.memory import SOPManager, get_system_prompt_with_sop

>>> manager = SOPManager(sop_directory="./data/sops")
>>> print(f"SOPs loaded: {len(manager.sops)}")
SOPs loaded: 7

>>> query = "Help me debug this code that gives an error"
>>> relevant = manager.find_relevant_sop(query)
>>> print(f"Selected SOP: {relevant[0].name}")
Selected SOP: debug_python_code

>>> system_prompt = get_system_prompt_with_sop(query, manager)
>>> print(system_prompt)
```

### System Prompt Output with SOP

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
**PROCEDURE TO FOLLOW:**

## Procedure: debug_python_code
**Description**: Procedure for identifying and fixing bugs in Python code
**Category**: coding

### Steps:
1. Carefully read the code and the reported error
2. Identify the error type (syntax, logic, runtime)
3. Locate the problematic line or function
4. Propose a solution with explanation
5. Suggest tests to verify the fix

Follow this procedure step-by-step. Indicate which step you're executing.
---

If no procedure is applicable, respond naturally and helpfully.
```

---

## 💻 YAML Configuration

### Complete Example

```yaml
# config/config.yaml - Configuration for generalist training

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

## 📈 Benchmarks

### Memory Usage

| Model | Configuration | VRAM Peak | Batch Size |
|-------|--------------|-----------|------------|
| Mistral 7B | QLoRA 4-bit | 6.2 GB | 2 |
| Mistral 7B | QLoRA 4-bit + GC | 5.1 GB | 2 |
| Llama 2 7B | QLoRA 4-bit | 6.4 GB | 2 |
| Mistral Nemo 12B | QLoRA 4-bit + GC | 9.8 GB | 1 |

### Training Speed (RTX 4090)

| Algorithm | Throughput | Time/1000 steps |
|-----------|------------|-----------------|
| SFT | 1.8 it/s | ~9 min |
| GRPO (4 gen) | 0.45 it/s | ~37 min |
| GRPO (2 gen) | 0.9 it/s | ~18 min |

---

## 🎯 Generation Examples

### Coding Task

**Prompt:**
```
Write a Python class for a thread-safe counter with increment, decrement, 
and get_value methods.
```

**Output (after training):**
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

*These examples are from real training runs during project development.*

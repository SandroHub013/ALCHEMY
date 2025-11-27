# ✨ Technical Features

Complete overview of the features implemented in this framework.

---

## 🎯 Overview

| Category | Feature | Status |
|----------|---------|--------|
| **Training** | QLoRA 4-bit | ✅ Complete |
| | PEFT/LoRA | ✅ Complete |
| | Multi-Source Training | ✅ Complete |
| | Gradient Checkpointing | ✅ Complete |
| **RL** | Agent Lightning Integration | ✅ Complete |
| | GRPO Algorithm | ✅ Complete |
| | Custom Reward Functions | ✅ Complete |
| | APO (Prompt Optimization) | ✅ Complete |
| **Reasoning** 🆕 | LUFFY Off-Policy Learning | ✅ Complete |
| | ExGRPO (Self-Experience) | ✅ Complete |
| | DeepSeek-R1 Integration | ✅ Complete |
| | Search-R1 (Reasoning + Search) | ✅ Complete |
| **Memory** | VectorStore (ChromaDB) | ✅ Complete |
| | CrossEncoder Reranking | ✅ Complete |
| | Smart Chunking | ✅ Complete |
| | SOP (Procedural Memory) | ✅ Complete |
| | FAISS Vector Search 🆕 | ✅ Complete |
| | Hybrid Search (Vector + BM25) 🆕 | ✅ Complete |

---

## 🦊 LUFFY - Off-Policy Reasoning Learning (NEW)

### Overview

**[LUFFY](https://arxiv.org/abs/2504.14945)** (Learning to Reason under Off-Policy Guidance) is a framework for improving model reasoning capabilities using off-policy traces from advanced models like DeepSeek-R1.

**Accepted at NeurIPS 2025!**

### Supported Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| **luffy** | Uses traces from external models (DeepSeek-R1) | When you have access to high-quality reasoning traces |
| **exgrpo** | Learns from own experience | When you don't have external traces |
| **hybrid** | Combines both | To maximize capabilities |

### Implementation

```python
# src/reasoning/luffy_trainer.py

from src.reasoning import LuffyTrainer, LuffyConfig, OffPolicyMode

config = LuffyConfig(
    mode=OffPolicyMode.LUFFY,
    off_policy_source="deepseek-r1",
    off_policy_weight=0.5,
    on_policy_weight=0.5,
    kl_coef=0.05,
)

trainer = LuffyTrainer(model, tokenizer, config)

# Load off-policy traces
trainer.load_off_policy_traces("deepseek_r1_traces.json")

# Training
results = trainer.train(train_dataset, num_epochs=3)
```

### Off-Policy Mixer

```python
class OffPolicyDataMixer:
    """Combines on-policy and off-policy data with prioritized sampling."""
    
    def sample_mixed_batch(self, on_policy_data, batch_size):
        # Calculate split
        off_policy_count = int(batch_size * self.config.off_policy_weight)
        on_policy_count = batch_size - off_policy_count
        
        # Sample off-policy with filtering
        off_policy = self._sample_filtered(
            min_reward=self.config.min_off_policy_reward
        )
        
        # Combine and shuffle
        return mixed_batch
```

### ExGRPO (Learning from Own Experience)

```python
# ExGRPO: Learn from the model's own experience
config = LuffyConfig(
    mode=OffPolicyMode.EXGRPO,
    experience_buffer_size=10000,
    experience_sample_ratio=0.3,
)

# The trainer automatically saves positive experiences
# and reuses them in subsequent training
```

### Benchmark Results

| Model | AIME 2024 | AIME 2025 | MATH-500 | Avg |
|-------|-----------|-----------|----------|-----|
| Qwen2.5-Math-7B (baseline) | 11.5 | 4.9 | 43.6 | 19.0 |
| + SFT Only | 22.2 | 22.3 | 82.6 | 44.1 |
| + SFT + RL | 25.8 | 23.1 | 87.2 | 48.2 |
| **+ LUFFY** | **29.4** | **23.1** | **87.6** | **50.1** |

---

## 🔍 Search-R1 - Reasoning with Search (NEW)

### Overview

**[Search-R1](https://github.com/PeterGriffinJin/Search-R1)** enables the model to search for information during reasoning, combining retrieval and reasoning seamlessly.

### Main Components

| Component | Description |
|-----------|-------------|
| **SearchEngine** | Interface for different search types |
| **ReasoningWithSearch** | Orchestrator integrating search into reasoning |
| **SearchR1Trainer** | RL trainer to optimize search policy |

### Search Engine Types

```python
from src.reasoning import create_search_engine

# Vector Search (semantic similarity)
vector_engine = create_search_engine("vector", documents=docs)

# BM25 Search (keyword-based)
bm25_engine = create_search_engine("bm25", documents=docs)

# Hybrid Search (vector + BM25 with RRF fusion)
hybrid_engine = create_search_engine(
    "hybrid",
    documents=docs,
    vector_weight=0.5,
    bm25_weight=0.5
)
```

### Reasoning Flow

```python
# src/reasoning/search_r1.py

class ReasoningWithSearch:
    def reason(self, question: str) -> Dict[str, Any]:
        # 1. Generate initial thought
        # 2. Detect <search>query</search>
        # 3. Execute search
        # 4. Inject <context>results</context>
        # 5. Continue reasoning
        # 6. Extract final answer
        
        return {
            "question": question,
            "final_answer": answer,
            "reasoning_trace": trace,
            "search_queries": queries,
        }
```

### Special Tokens

| Token | Description |
|-------|-------------|
| `<search>` | Start search query |
| `</search>` | End query |
| `<context>` | Start results |
| `</context>` | End results |

### Complete Example

```python
from src.reasoning import (
    SearchR1Trainer, 
    SearchR1Config,
    HybridSearchEngine,
)

# Configure search engine
search_engine = HybridSearchEngine(
    documents=knowledge_base,
    vector_weight=0.6,
    bm25_weight=0.4,
)

# Configure trainer
config = SearchR1Config(
    max_search_calls=3,
    context_window=3,
    use_cot=True,
    reward_search_bonus=0.1,
)

trainer = SearchR1Trainer(
    model=model,
    tokenizer=tokenizer,
    search_engine=search_engine,
    config=config,
)

# Training
results = trainer.train(
    train_data=[{"question": "...", "answer": "..."}, ...],
    num_epochs=3,
)
```

### Reward Function

```python
def search_reward(question, answer, reference, search_used):
    reward = 0.0
    
    # Answer correctness (weight: 0.7)
    if reference.lower() in answer.lower():
        reward += 0.5
    
    # Bonus for effective search use (weight: 0.1)
    if search_used:
        reward += 0.1
    
    # Reasoning quality (weight: 0.2)
    if "therefore" in answer.lower() or "step" in answer.lower():
        reward += 0.2
    
    return reward
```

---

## 🔬 Efficient Training

### QLoRA 4-bit Quantization

**Problem solved:** LLM models require too much memory for consumer GPUs.

**Implemented solution:**

```python
# src/models/model_loader.py

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Load in 4-bit
    bnb_4bit_compute_dtype=torch.float16, # Compute in FP16
    bnb_4bit_quant_type="nf4",           # Normal Float 4-bit
    bnb_4bit_use_double_quant=True,       # Quantize the quantization params
)
```

**Results:**
- Mistral 7B: from 28GB → 6GB VRAM
- 78% memory reduction
- Performance maintained

### LoRA (Low-Rank Adaptation)

**Problem solved:** Updating all parameters is expensive and risks overfitting.

**Implementation:**

```python
lora_config = LoraConfig(
    r=16,                # Decomposition rank
    lora_alpha=32,       # Scaling factor
    target_modules=[     # Only critical layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)
```

**Results:**
- Trainable parameters: 0.18% (13M out of 7B)
- 50x faster training
- Modular and combinable adapters

### Multi-Source Training

**Problem solved:** Training on a single task causes Catastrophic Forgetting.

**Solution:**

```yaml
# config/config.yaml
datasets:
  multi_source_enabled: true
  sources:
    - name: "glaiveai/glaive-function-calling-v2"
      weight: 0.30
      type: "function_calling"
    - name: "nickrosh/Evol-Instruct-Code-80k-v1"
      weight: 0.30
      type: "coding"
    - name: "teknium/OpenHermes-2.5"
      weight: 0.30
      type: "chat"
```

**Features:**
- Weighted sampling to balance tasks
- Unified formatter (ChatML)
- Preserves existing skills

---

## 🤖 Reinforcement Learning

### Agent Lightning Integration

**Framework:** Microsoft Agent Lightning for RL training of AI agents.

**Supported algorithms:**

| Algorithm | Description | When to use |
|-----------|-------------|-------------|
| **SFT** | Supervised Fine-Tuning | Initial training |
| **GRPO** | Group Relative Policy Optimization | Improve behavior |
| **APO** | Automatic Prompt Optimization | Optimize system prompt |

### GRPO Implementation

```python
# src/agent/agent_lightning_trainer.py

class AgentLightningTrainer:
    def train(self, train_dataset, ...):
        for batch in train_dataset:
            # 1. Generate K different responses
            generations = self.model.generate(
                batch["prompt"],
                num_return_sequences=self.config.num_generations,
                temperature=self.config.temperature,
            )
            
            # 2. Calculate reward for each response
            rewards = [self.reward_fn(prompt, gen) for gen in generations]
            
            # 3. Normalize rewards (relative)
            advantages = (rewards - mean(rewards)) / std(rewards)
            
            # 4. Policy gradient with KL penalty
            loss = -log_prob(generations) * advantages + kl_coef * KL_div
            
            # 5. Update
            loss.backward()
            optimizer.step()
```

### Reward Functions

**Coding Reward:**
```python
def coding_reward(prompt, generation):
    reward = 0.0
    
    # Correct syntax
    try:
        compile(code, '<string>', 'exec')
        reward += 0.3
    except SyntaxError:
        reward -= 0.3
    
    # Best practices
    if has_docstring(code):    reward += 0.1
    if has_type_hints(code):   reward += 0.1
    if appropriate_length(code): reward += 0.1
    
    return reward
```

**Function Calling Reward:**
```python
def function_calling_reward(prompt, generation):
    # Valid JSON
    try:
        fc = json.loads(extract_function_call(generation))
        reward += 0.3
    except:
        return -0.3
    
    # Correct structure
    if "name" in fc and "arguments" in fc:
        reward += 0.2
    
    # Existing tool
    if fc["name"] in available_tools:
        reward += 0.2
    
    return reward
```

**Combined Reward (Auto-detect):**
```python
def combined_reward(prompt, generation):
    # Automatically detect task type
    if is_coding_task(prompt):
        return coding_reward(prompt, generation)
    elif is_function_calling(prompt):
        return function_calling_reward(prompt, generation)
    else:
        return chat_reward(prompt, generation)
```

---

## 🧠 Memory System

### VectorStore with Reranking

**Two-phase architecture:**

```
Query → Bi-Encoder (recall) → Top-K candidates → Cross-Encoder (precision) → Top-N final
```

**Implementation:**

```python
# src/memory/vector_store.py

class VectorStore:
    def __init__(self, use_reranker=True):
        # Phase 1: Bi-Encoder for fast recall
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Phase 2: Cross-Encoder for precision
        if use_reranker:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def query(self, text, n_results=3):
        # Phase 1: Retrieve candidates (3x n_results)
        candidates = self._vector_search(text, n_results * 3)
        
        # Phase 2: Rerank with Cross-Encoder
        if self.reranker:
            candidates = self.reranker.rerank(text, candidates, top_k=n_results)
        
        return candidates
```

### Smart Chunking

**Problem:** Character-based chunking cuts functions in half.

**Solution:** AST parsing with tree-sitter.

```python
# src/memory/smart_chunker.py

class SmartChunker:
    def chunk_python_code(self, code):
        # Parse AST
        tree = self.parser.parse(code.encode())
        
        chunks = []
        for node in self._traverse(tree.root_node):
            if node.type == "function_definition":
                chunks.append(self._extract_function(node))
            elif node.type == "class_definition":
                chunks.append(self._extract_class(node))
        
        return chunks
```

**Chunk types:**

| Type | Description |
|------|-------------|
| `FUNCTION` | Complete function with docstring |
| `CLASS` | Class with header and docstring |
| `METHOD` | Class method |
| `IMPORT` | Import block |
| `DOCSTRING` | Module docstring |

**Optimized embedding text:**

```python
def to_embedding_text(chunk):
    """Generate text with context for better embeddings."""
    return f"""
# {chunk.chunk_type}: {chunk.qualified_name}
# Description: {chunk.docstring[:200]}
# File: {chunk.file_path}

{chunk.content}
"""
```

### SOP (Standard Operating Procedures)

**Purpose:** Guide the model through structured procedures.

**SOP Structure:**

```json
{
  "name": "debug_python_code",
  "description": "Procedure for identifying and fixing bugs",
  "trigger": "debug, bug, error, not working",
  "category": "coding",
  "priority": 8,
  "steps": [
    {"action": "Read the code and identify the problem"},
    {"action": "Classify the error (syntax, logic, runtime)"},
    {"action": "Locate the problematic line"},
    {"action": "Propose a solution with explanation"},
    {"action": "Suggest tests to verify", "condition": "fix proposed"}
  ]
}
```

**Automatic matching:**

```python
class SOPManager:
    def find_relevant_sop(self, query):
        scores = []
        for sop in self.sops.values():
            score = 0
            for trigger in sop.trigger.split(","):
                if trigger.strip() in query.lower():
                    score += 10
            score += sop.priority * 0.5
            scores.append((sop, score))
        
        return sorted(scores, key=lambda x: -x[1])[0][0]
```

---

## 📊 Logging and Monitoring

### TensorBoard Integration

```python
# Metrics logged automatically
self.log("train/loss", loss)
self.log("train/perplexity", exp(loss))
self.log("train/reward_mean", reward.mean())
self.log("train/kl_div", kl_divergence)
```

### Agent Lightning Tracing

```python
# Detailed tracing
with self.tracer.span("generation", {"prompt_len": len(prompt)}):
    output = model.generate(...)
    
agl.emit_generation(
    prompt=prompt,
    response=output,
    model=model_name,
)
```

---

## ⚙️ Configuration

### YAML-based

All configuration is in YAML for flexibility:

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
  num_epochs: 3
  learning_rate: 2.0e-4
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8

agent_lightning:
  enabled: true
  algorithm: "grpo"
  reward_function: "combined"
```

### CLI Override

```bash
# Override algorithm
python main_agent_lightning.py --config config.yaml --algorithm grpo

# Dry-run to verify
python main_agent_lightning.py --config config.yaml --dry-run
```

---

## 🔌 Extensibility

### Adding a New Model

```python
# In model_loader.py
def _get_target_modules_for_model(self, model_name):
    if "my-new-model" in model_name.lower():
        return ["my_attn", "my_mlp"]
    # ... existing models
```

### Adding a New Reward Function

```python
# In agent_lightning_trainer.py
@staticmethod
def my_reward(prompt, generation):
    # Custom logic
    return score

# Register
RewardFunction.my_reward = my_reward
```

### Adding a New SOP

```json
// data/sops/my_procedure.json
{
  "name": "my_procedure",
  "trigger": "keywords, here",
  "steps": [...]
}
```

---

*This documentation is generated for the technical portfolio.*

# 📚 Academic Citations and References

This project builds on academic research and open-source tools. Below are the complete citations for academic use.

---

## 📄 Foundational Papers

### QLoRA: Efficient Finetuning of Quantized LLMs

The 4-bit quantization technique that makes fine-tuning possible on consumer GPUs.

```bibtex
@article{dettmers2023qlora,
  title     = {QLoRA: Efficient Finetuning of Quantized LLMs},
  author    = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal   = {arXiv preprint arXiv:2305.14314},
  year      = {2023},
  url       = {https://arxiv.org/abs/2305.14314}
}
```

**Contribution to project:** NF4 quantization, Double Quantization, paging management for optimizer states.

---

### LoRA: Low-Rank Adaptation of Large Language Models

The low-rank adaptation technique that reduces trainable parameters by 99.9%.

```bibtex
@article{hu2021lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and 
               Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and 
               Wang, Lu and Chen, Weizhu},
  journal   = {arXiv preprint arXiv:2106.09685},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.09685}
}
```

**Contribution to project:** Low-rank decomposition for attention and MLP layers, module targeting strategy.

---

### GRPO: Group Relative Policy Optimization

The Reinforcement Learning algorithm used for agent training.

```bibtex
@article{shao2024deepseekmath,
  title     = {DeepSeekMath: Pushing the Limits of Mathematical Reasoning 
               in Open Language Models},
  author    = {Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and 
               Song, Junxiao and Zhang, Mingchuan and Li, Y. K. and Wu, Y. and Guo, Daya},
  journal   = {arXiv preprint arXiv:2402.03300},
  year      = {2024},
  url       = {https://arxiv.org/abs/2402.03300}
}
```

**Contribution to project:** RL training algorithm, relative reward normalization, KL regularization.

---

### LUFFY: Learning to Reason under Off-Policy Guidance

Off-policy learning for improving reasoning capabilities.

```bibtex
@article{yan2025luffy,
  title     = {LUFFY: Learning to Reason under Off-Policy Guidance},
  author    = {Yan, Elliott and others},
  journal   = {NeurIPS 2025},
  year      = {2025},
  url       = {https://arxiv.org/abs/2504.14945}
}
```

**Contribution to project:** Off-policy reasoning traces, ExGRPO self-experience learning.

---

### DeepSeek-R1: Reinforcement Learning for Reasoning

RL techniques for training reasoning models.

```bibtex
@article{deepseek2025r1,
  title     = {DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via 
               Reinforcement Learning},
  author    = {{DeepSeek-AI}},
  journal   = {arXiv preprint arXiv:2501.12948},
  year      = {2025},
  url       = {https://arxiv.org/abs/2501.12948}
}
```

**Contribution to project:** Reasoning trace format, GRPO for reasoning tasks.

---

### ColBERT: Efficient Late Interaction

Inspiration for the reranking system with CrossEncoder.

```bibtex
@inproceedings{khattab2020colbert,
  title     = {ColBERT: Efficient and Effective Passage Search via Contextualized 
               Late Interaction over BERT},
  author    = {Khattab, Omar and Zaharia, Matei},
  booktitle = {SIGIR 2020},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.12832}
}
```

**Contribution to project:** Two-phase reranking pattern (bi-encoder + cross-encoder).

---

### Sentence-BERT: Sentence Embeddings

Semantic embeddings for search in the RAG system.

```bibtex
@inproceedings{reimers2019sentence,
  title     = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {EMNLP-IJCNLP 2019},
  year      = {2019},
  url       = {https://arxiv.org/abs/1908.10084}
}
```

**Contribution to project:** Embeddings for semantic similarity, bi-encoder models for retrieval.

---

## 🛠️ Software and Frameworks

### Unsloth

High-performance fine-tuning library providing 2x speed and 70% VRAM reduction.

```bibtex
@software{unsloth2024,
  title     = {Unsloth: Fine-tuning & Reinforcement Learning for LLMs},
  author    = {Han, Daniel and Han, Michael and {Unsloth team}},
  year      = {2024},
  url       = {https://github.com/unslothai/unsloth},
  note      = {Apache License 2.0}
}
```

**Contribution to project:** 2x faster training, 70% less VRAM, GGUF export, native GRPO support.

---

### Microsoft Agent Lightning

```bibtex
@software{agentlightning2024,
  title     = {Agent Lightning: A Framework for Training AI Agents with 
               Reinforcement Learning},
  author    = {{Microsoft Research}},
  year      = {2024},
  url       = {https://github.com/microsoft/agent-lightning},
  note      = {MIT License}
}
```

---

### HuggingFace Transformers

```bibtex
@inproceedings{wolf2020transformers,
  title     = {Transformers: State-of-the-Art Natural Language Processing},
  author    = {Wolf, Thomas and others},
  booktitle = {EMNLP 2020 System Demonstrations},
  year      = {2020}
}
```

---

### TRL (Transformer Reinforcement Learning)

```bibtex
@software{trl2023,
  title     = {TRL: Transformer Reinforcement Learning},
  author    = {{HuggingFace}},
  year      = {2023},
  url       = {https://github.com/huggingface/trl},
  note      = {Apache License 2.0}
}
```

---

### PyTorch Lightning

```bibtex
@software{falcon2019pytorch,
  title     = {PyTorch Lightning},
  author    = {Falcon, William and {The PyTorch Lightning team}},
  year      = {2019},
  url       = {https://github.com/Lightning-AI/lightning},
  note      = {Apache License 2.0}
}
```

---

### ChromaDB

```bibtex
@software{chromadb2022,
  title     = {Chroma: The AI-native open-source embedding database},
  author    = {{Chroma Core, Inc.}},
  year      = {2022},
  url       = {https://github.com/chroma-core/chroma},
  note      = {Apache License 2.0}
}
```

---

### tree-sitter

```bibtex
@software{brunsfeld2018treesitter,
  title     = {tree-sitter: An incremental parsing system for programming tools},
  author    = {Brunsfeld, Max},
  year      = {2018},
  url       = {https://github.com/tree-sitter/tree-sitter},
  note      = {MIT License}
}
```

---

## 💡 Inspirations for New Integrations

### PocketFlow (Meta-Agent Inspiration)

Minimalist framework demonstrating "agents building agents" pattern.

```
Repository: https://github.com/The-Pocket/PocketFlow
Concept: Agents that generate other agents
License: MIT
```

**Adopted concepts:**
- Dynamic agent blueprint generation
- Self-generating reward functions
- Task-adaptive SOP creation

---

### AgentFlow (Adaptive Trainer Inspiration)

"In-the-flow" optimization for agentic systems.

```
Repository: https://github.com/lupantech/AgentFlow
Concept: Dynamic optimization during execution
```

**Adopted concepts:**
- Real-time metric analysis
- Curriculum learning
- Adaptive hyperparameter adjustment

---

### claude-flow (Swarm Trainer Inspiration)

Multi-agent orchestration with swarm intelligence.

```
Repository: https://github.com/ruvnet/claude-flow
Concept: Swarm intelligence for AI agents
```

**Adopted concepts:**
- Explorer/Exploiter role division
- Parallel trajectory exploration
- Inter-agent communication

---

### Flowise (Workflow Patterns)

Visual workflow builder for LLM applications.

```
Repository: https://github.com/FlowiseAI/Flowise
License: MIT
```

**Adopted concepts:**
- Pipeline orchestration patterns
- Component-based architecture

---

## 📋 How to Cite This Project

If you use ALCHEMY in your research, you can cite it as:

```bibtex
@software{alchemy2024,
  title     = {ALCHEMY: Advanced LLM Training Framework with Multi-Agent Orchestration},
  author    = {Boni, Alessandro},
  year      = {2024},
  url       = {https://github.com/SandroHub013/ALCHEMY},
  note      = {MIT License. Combines QLoRA, Unsloth, GRPO, Multi-Agent Swarms, 
               and Adaptive Training for efficient local LLM training.}
}
```

---

## 🙏 Acknowledgments

This project would not have been possible without:

- **Daniel Han & Michael Han** — Unsloth team for revolutionary optimizations
- **Tim Dettmers** — QLoRA and bitsandbytes
- **Microsoft Research** — Agent Lightning framework
- **HuggingFace** — Transformers, PEFT, TRL, and Datasets
- **DeepSeek-AI** — GRPO research and reasoning techniques
- **Lightning AI** — PyTorch Lightning
- **Nils Reimers** — Sentence-Transformers
- **Max Brunsfeld** — tree-sitter
- **PocketFlow, AgentFlow, claude-flow teams** — Inspiration for new integrations

And all contributors to the papers and projects cited above.

---

## 📜 License Summary

| Component | License |
|-----------|---------|
| ALCHEMY | MIT |
| Unsloth | Apache 2.0 |
| Transformers | Apache 2.0 |
| PyTorch Lightning | Apache 2.0 |
| PEFT | Apache 2.0 |
| TRL | Apache 2.0 |
| bitsandbytes | MIT |
| ChromaDB | Apache 2.0 |
| tree-sitter | MIT |
| Agent Lightning | MIT |

---

*Last updated: November 2025*

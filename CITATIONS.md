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
  url       = {https://arxiv.org/abs/2305.14314},
  abstract  = {We present QLoRA, an efficient finetuning approach that reduces memory 
               usage enough to finetune a 65B parameter model on a single 48GB GPU 
               while preserving full 16-bit finetuning task performance. QLoRA 
               backpropagates gradients through a frozen, 4-bit quantized pretrained 
               language model into Low Rank Adapters (LoRA).}
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
  url       = {https://arxiv.org/abs/2106.09685},
  abstract  = {We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained 
               model weights and injects trainable rank decomposition matrices into 
               each layer of the Transformer architecture, greatly reducing the number 
               of trainable parameters for downstream tasks.}
}
```

**Contribution to project:** Low-rank decomposition for attention and MLP layers, module targeting strategy.

---

### GRPO: Group Relative Policy Optimization

The Reinforcement Learning algorithm used by Agent Lightning for agent training.

```bibtex
@article{shao2024deepseekmath,
  title     = {DeepSeekMath: Pushing the Limits of Mathematical Reasoning 
               in Open Language Models},
  author    = {Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and 
               Song, Junxiao and Zhang, Mingchuan and Li, Y. K. and Wu, Y. and Guo, Daya},
  journal   = {arXiv preprint arXiv:2402.03300},
  year      = {2024},
  url       = {https://arxiv.org/abs/2402.03300},
  abstract  = {We introduce DeepSeekMath 7B, which achieves an impressive score of 
               51.7% on the competition-level MATH benchmark without relying on 
               external toolkits and voting techniques. In this paper, we also 
               introduce Group Relative Policy Optimization (GRPO), a variant of 
               Proximal Policy Optimization (PPO).}
}
```

**Contribution to project:** RL training algorithm, relative reward normalization, KL regularization.

---

### Sentence-BERT: Sentence Embeddings

Semantic embeddings for search in the RAG system.

```bibtex
@inproceedings{reimers2019sentence,
  title     = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods 
               in Natural Language Processing and the 9th International 
               Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages     = {3982--3992},
  year      = {2019},
  publisher = {Association for Computational Linguistics},
  url       = {https://arxiv.org/abs/1908.10084}
}
```

**Contribution to project:** Embeddings for semantic similarity, bi-encoder models for retrieval.

---

### ColBERT: Efficient Late Interaction

Inspiration for the reranking system with CrossEncoder.

```bibtex
@inproceedings{khattab2020colbert,
  title     = {ColBERT: Efficient and Effective Passage Search via Contextualized 
               Late Interaction over BERT},
  author    = {Khattab, Omar and Zaharia, Matei},
  booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on 
               Research and Development in Information Retrieval},
  pages     = {39--48},
  year      = {2020},
  publisher = {ACM},
  url       = {https://arxiv.org/abs/2004.12832}
}
```

**Contribution to project:** Two-phase reranking pattern (bi-encoder + cross-encoder).

---

### Transformer Architecture

The architecture underlying all modern LLMs.

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and 
               Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and 
               Kaiser, Lukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017},
  publisher = {Curran Associates, Inc.},
  url       = {https://arxiv.org/abs/1706.03762}
}
```

---

## 🛠️ Software and Frameworks

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

### HuggingFace Transformers

```bibtex
@inproceedings{wolf2020transformers,
  title     = {Transformers: State-of-the-Art Natural Language Processing},
  author    = {Wolf, Thomas and Debut, Lysandre and Sanh, Victor and 
               Chaumond, Julien and Delangue, Clement and Moi, Anthony and 
               Cistac, Pierric and Rault, Tim and Louf, R{\'e}mi and 
               Funtowicz, Morgan and Davison, Joe and Shleifer, Sam and 
               von Platen, Patrick and Ma, Clara and Jernite, Yacine and 
               Plu, Julien and Xu, Canwen and Le Scao, Teven and 
               Gugger, Sylvain and Drame, Mariama and Lhoest, Quentin and 
               Rush, Alexander},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods 
               in Natural Language Processing: System Demonstrations},
  pages     = {38--45},
  year      = {2020},
  publisher = {Association for Computational Linguistics}
}
```

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

## 💡 Inspirations and References

### osgrep - Semantic Code Search

The smart chunking system is inspired by osgrep's approach to semantic code search.

```
Repository: https://github.com/Ryandonofrio3/osgrep
Author: Ryan Donofrio
License: MIT
```

**Adopted concepts:**
- AST-aware chunking with tree-sitter
- Reranking with CrossEncoder
- Preservation of semantic boundaries

---

## 📋 How to Cite This Project

If you use this project in your research, you can cite it as:

```bibtex
@software{llm_finetuning_agent_lightning,
  title     = {LLM Fine-tuning with Agent Lightning: A Framework for Local 
               Training with Reinforcement Learning and RAG},
  author    = {[Your Name]},
  year      = {2024},
  url       = {https://github.com/[username]/llm-finetuning-agent-lightning},
  note      = {MIT License. Combines QLoRA, Agent Lightning GRPO, 
               and RAG for efficient local LLM training.}
}
```

---

## 🙏 Acknowledgments

This project would not have been possible without the work of the open-source community:

- **Tim Dettmers** and team for QLoRA and bitsandbytes
- **Microsoft Research** for Agent Lightning
- **HuggingFace** for Transformers, PEFT, and Datasets
- **Lightning AI** for PyTorch Lightning
- **Nils Reimers** for Sentence-Transformers
- **Max Brunsfeld** for tree-sitter

And all contributors to the papers and projects cited above.

---

*Last updated: November 2024*

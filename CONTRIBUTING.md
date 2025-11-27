# 🤝 Contribution Guide

Thank you for your interest in contributing to this project! This guide will help you get started.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Environment Setup](#development-environment-setup)
- [Code Style](#code-style)
- [Pull Requests](#pull-requests)
- [Reporting Bugs](#reporting-bugs)
- [Proposing Features](#proposing-features)

---

## 📜 Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/). By participating, you agree to uphold these principles:

- 🤝 **Be respectful** - Treat everyone with respect and professionalism
- 💡 **Be constructive** - Criticize the code, not the person
- 🌍 **Be inclusive** - We welcome contributions from everyone
- 📚 **Be patient** - Not everyone has the same level of experience

---

## 🚀 How to Contribute

### Welcome Contribution Types

| Type | Description |
|------|-------------|
| 🐛 **Bug Fix** | Fixes for documented bugs |
| ✨ **Features** | New functionality (discuss first in an Issue) |
| 📖 **Documentation** | Documentation improvements |
| 🧪 **Tests** | New tests or coverage improvements |
| 🔧 **Refactoring** | Code improvements without changing behavior |
| 🌍 **Translations** | Documentation translation |

### Workflow

```
1. Fork the repository
         │
         ▼
2. Create a branch
   git checkout -b feature/feature-name
         │
         ▼
3. Make your changes
         │
         ▼
4. Run tests
   pytest tests/
         │
         ▼
5. Commit with clear message
   git commit -m "feat: add support for X"
         │
         ▼
6. Push the branch
   git push origin feature/feature-name
         │
         ▼
7. Open a Pull Request
```

---

## 💻 Development Environment Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (optional, for full testing)
- Git

### Installation

```bash
# 1. Clone your fork
git clone https://github.com/YOUR-USERNAME/llm-finetuning-agent-lightning.git
cd llm-finetuning-agent-lightning

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# 3. Install development dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check linting
ruff check src/

# Check types
mypy src/
```

---

## 📝 Code Style

### Python

We follow [PEP 8](https://peps.python.org/pep-0008/) with some customizations:

```python
# ✅ Good
def calculate_reward(
    prompt: str,
    generation: str,
    reference: Optional[str] = None,
) -> float:
    """
    Calculate the reward for a generation.
    
    Args:
        prompt: The original prompt.
        generation: The generated response.
        reference: Reference response (optional).
        
    Returns:
        Reward value between -1.0 and 1.0.
        
    Raises:
        ValueError: If prompt is empty.
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    
    reward = 0.0
    
    # Calculation logic...
    
    return max(-1.0, min(1.0, reward))


# ❌ Avoid
def calc_rew(p, g, r=None):
    if not p: raise ValueError()
    rew = 0
    # ...
    return max(-1, min(1, rew))
```

### Conventions

| Element | Style | Example |
|---------|-------|---------|
| Functions | `snake_case` | `calculate_reward()` |
| Classes | `PascalCase` | `VectorStore` |
| Constants | `UPPER_SNAKE` | `MAX_CHUNK_SIZE` |
| Variables | `snake_case` | `embedding_model` |
| Modules | `snake_case` | `vector_store.py` |

### Docstrings

We use Google format:

```python
def query(
    self,
    text: str,
    n_results: int = 3,
    use_reranker: Optional[bool] = None,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Search for documents most similar to a query.
    
    If the reranker is enabled, retrieves more initial results and then
    reorders them by relevance using CrossEncoder.
    
    Args:
        text: Query text.
        n_results: Number of results to return.
        use_reranker: Override to use/not use reranker.
            None = use instance default.
            
    Returns:
        List of tuples (document, score, metadata).
        Higher score = more relevant.
        
    Raises:
        ValueError: If n_results < 1.
        
    Example:
        >>> store = VectorStore()
        >>> store.add_documents(["Python is a language..."])
        >>> results = store.query("What is Python?")
        >>> print(results[0][0])  # First document
    """
```

### Type Hints

We use type hints everywhere:

```python
from typing import Optional, Dict, List, Any, Tuple, Union

def process_data(
    items: List[str],
    config: Dict[str, Any],
    max_items: Optional[int] = None,
) -> Tuple[List[str], int]:
    ...
```

---

## 🔄 Pull Requests

### Checklist

Before opening a PR, verify:

- [ ] Code follows style conventions
- [ ] Tests pass (`pytest tests/`)
- [ ] I added tests for new functionality
- [ ] Documentation is updated
- [ ] Commits follow conventions

### Commit Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting (no logic) |
| `refactor` | Refactoring |
| `test` | Add/modify tests |
| `chore` | Maintenance |

**Examples:**

```bash
feat(memory): add smart chunking with tree-sitter

fix(training): handle empty batch in validation step

docs(readme): add benchmark results section

refactor(vector_store): extract reranker to separate class
```

### PR Template

```markdown
## Description

Brief description of the changes.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## How was it tested?

Describe the tests performed.

## Checklist

- [ ] Code follows project style
- [ ] I performed self-review of my code
- [ ] I commented the code where necessary
- [ ] I updated the documentation
- [ ] Tests pass
- [ ] I added tests for new functionality
```

---

## 🐛 Reporting Bugs

### Bug Issue Template

```markdown
## Description

Clear description of the bug.

## How to Reproduce

1. Run '...'
2. With parameters '...'
3. See error

## Expected Behavior

What should happen.

## Actual Behavior

What happens instead.

## Environment

- OS: [e.g. Windows 10, Ubuntu 22.04]
- Python: [e.g. 3.10.12]
- PyTorch: [e.g. 2.1.0]
- CUDA: [e.g. 12.1]
- GPU: [e.g. RTX 4090]

## Log/Traceback

```python
# Paste error here
```

## Screenshots

If applicable.
```

---

## 💡 Proposing Features

### Before Proposing

1. **Search Issues** - It might already be proposed
2. **Consider scope** - Complex features require discussion
3. **Think about impact** - How does it affect existing users?

### Feature Issue Template

```markdown
## Problem/Motivation

Describe the problem this feature solves.

## Proposed Solution

Describe how you want it to work.

## Alternatives Considered

Other solutions you considered.

## Impact

- [ ] Breaking change
- [ ] New dependencies
- [ ] Config changes

## Implementation

Are you willing to implement it? Do you need help?
```

---

## 🙏 Acknowledgments

Every contribution, big or small, is appreciated. Thank you for making this project better!

---

*For questions, open an [Issue](https://github.com/SandroHub013/ALCHEMY/issues)!*

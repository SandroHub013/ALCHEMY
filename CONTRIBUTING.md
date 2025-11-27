# 🤝 Guida alla Contribuzione

Grazie per il tuo interesse nel contribuire a questo progetto! Questa guida ti aiuterà a iniziare.

## 📋 Tabella dei Contenuti

- [Codice di Condotta](#codice-di-condotta)
- [Come Contribuire](#come-contribuire)
- [Setup Ambiente di Sviluppo](#setup-ambiente-di-sviluppo)
- [Stile del Codice](#stile-del-codice)
- [Pull Request](#pull-request)
- [Segnalare Bug](#segnalare-bug)
- [Proporre Features](#proporre-features)

---

## 📜 Codice di Condotta

Questo progetto segue il [Contributor Covenant](https://www.contributor-covenant.org/). Partecipando, ti impegni a rispettare questi principi:

- 🤝 **Sii rispettoso** - Trattiamo tutti con rispetto e professionalità
- 💡 **Sii costruttivo** - Critica il codice, non la persona
- 🌍 **Sii inclusivo** - Accogliamo contributi da tutti
- 📚 **Sii paziente** - Non tutti hanno lo stesso livello di esperienza

---

## 🚀 Come Contribuire

### Tipi di Contributi Benvenuti

| Tipo | Descrizione |
|------|-------------|
| 🐛 **Bug Fix** | Correzioni di bug documentati |
| ✨ **Features** | Nuove funzionalità (discuti prima in una Issue) |
| 📖 **Documentazione** | Miglioramenti alla documentazione |
| 🧪 **Test** | Nuovi test o miglioramento copertura |
| 🔧 **Refactoring** | Miglioramenti al codice senza cambiare comportamento |
| 🌍 **Traduzioni** | Traduzione della documentazione |

### Workflow

```
1. Fork del repository
         │
         ▼
2. Crea un branch
   git checkout -b feature/nome-feature
         │
         ▼
3. Fai le modifiche
         │
         ▼
4. Esegui i test
   pytest tests/
         │
         ▼
5. Commit con messaggio chiaro
   git commit -m "feat: aggiungi supporto per X"
         │
         ▼
6. Push del branch
   git push origin feature/nome-feature
         │
         ▼
7. Apri una Pull Request
```

---

## 💻 Setup Ambiente di Sviluppo

### Requisiti

- Python 3.10+
- CUDA-capable GPU (opzionale, per test completi)
- Git

### Installazione

```bash
# 1. Clona il tuo fork
git clone https://github.com/TUO-USERNAME/llm-finetuning-agent-lightning.git
cd llm-finetuning-agent-lightning

# 2. Crea un ambiente virtuale
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
.\venv\Scripts\activate   # Windows

# 3. Installa dipendenze di sviluppo
pip install -e ".[dev]"

# 4. Installa pre-commit hooks
pre-commit install
```

### Verifica Installazione

```bash
# Esegui i test
pytest tests/ -v

# Controlla linting
ruff check src/

# Controlla types
mypy src/
```

---

## 📝 Stile del Codice

### Python

Seguiamo [PEP 8](https://peps.python.org/pep-0008/) con alcune personalizzazioni:

```python
# ✅ Buono
def calculate_reward(
    prompt: str,
    generation: str,
    reference: Optional[str] = None,
) -> float:
    """
    Calcola il reward per una generazione.
    
    Args:
        prompt: Il prompt originale.
        generation: La risposta generata.
        reference: Risposta di riferimento (opzionale).
        
    Returns:
        Valore di reward tra -1.0 e 1.0.
        
    Raises:
        ValueError: Se prompt è vuoto.
    """
    if not prompt:
        raise ValueError("Il prompt non può essere vuoto")
    
    reward = 0.0
    
    # Logica di calcolo...
    
    return max(-1.0, min(1.0, reward))


# ❌ Evita
def calc_rew(p, g, r=None):
    if not p: raise ValueError()
    rew = 0
    # ...
    return max(-1, min(1, rew))
```

### Convenzioni

| Elemento | Stile | Esempio |
|----------|-------|---------|
| Funzioni | `snake_case` | `calculate_reward()` |
| Classi | `PascalCase` | `VectorStore` |
| Costanti | `UPPER_SNAKE` | `MAX_CHUNK_SIZE` |
| Variabili | `snake_case` | `embedding_model` |
| Moduli | `snake_case` | `vector_store.py` |

### Docstring

Usiamo il formato Google:

```python
def query(
    self,
    text: str,
    n_results: int = 3,
    use_reranker: Optional[bool] = None,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Cerca i documenti più simili a una query.
    
    Se il reranker è abilitato, recupera più risultati iniziali e poi
    li riordina per rilevanza usando CrossEncoder.
    
    Args:
        text: Testo della query.
        n_results: Numero di risultati da restituire.
        use_reranker: Override per usare/non usare reranker.
            None = usa default dell'istanza.
            
    Returns:
        Lista di tuple (documento, score, metadata).
        Score più alto = più rilevante.
        
    Raises:
        ValueError: Se n_results < 1.
        
    Example:
        >>> store = VectorStore()
        >>> store.add_documents(["Python è un linguaggio..."])
        >>> results = store.query("Cos'è Python?")
        >>> print(results[0][0])  # Primo documento
    """
```

### Type Hints

Usiamo type hints ovunque:

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

## 🔄 Pull Request

### Checklist

Prima di aprire una PR, verifica:

- [ ] Il codice segue le convenzioni di stile
- [ ] I test passano (`pytest tests/`)
- [ ] Ho aggiunto test per le nuove funzionalità
- [ ] La documentazione è aggiornata
- [ ] I commit seguono le convenzioni

### Formato Commit

Usiamo [Conventional Commits](https://www.conventionalcommits.org/):

```
<tipo>(<scope>): <descrizione>

[corpo opzionale]

[footer opzionale]
```

**Tipi:**

| Tipo | Descrizione |
|------|-------------|
| `feat` | Nuova funzionalità |
| `fix` | Correzione bug |
| `docs` | Solo documentazione |
| `style` | Formattazione (no logica) |
| `refactor` | Refactoring |
| `test` | Aggiunta/modifica test |
| `chore` | Maintenance |

**Esempi:**

```bash
feat(memory): add smart chunking with tree-sitter

fix(training): handle empty batch in validation step

docs(readme): add benchmark results section

refactor(vector_store): extract reranker to separate class
```

### Template PR

```markdown
## Descrizione

Breve descrizione delle modifiche.

## Tipo di Cambiamento

- [ ] Bug fix
- [ ] Nuova feature
- [ ] Breaking change
- [ ] Documentazione

## Come è stato testato?

Descrivi i test eseguiti.

## Checklist

- [ ] Il codice segue lo stile del progetto
- [ ] Ho eseguito self-review del mio codice
- [ ] Ho commentato il codice dove necessario
- [ ] Ho aggiornato la documentazione
- [ ] I test passano
- [ ] Ho aggiunto test per le nuove funzionalità
```

---

## 🐛 Segnalare Bug

### Template Issue Bug

```markdown
## Descrizione

Descrizione chiara del bug.

## Come Riprodurre

1. Esegui '...'
2. Con parametri '...'
3. Vedi errore

## Comportamento Atteso

Cosa dovrebbe succedere.

## Comportamento Attuale

Cosa succede invece.

## Ambiente

- OS: [es. Windows 10, Ubuntu 22.04]
- Python: [es. 3.10.12]
- PyTorch: [es. 2.1.0]
- CUDA: [es. 12.1]
- GPU: [es. RTX 4090]

## Log/Traceback

```python
# Incolla qui l'errore
```

## Screenshot

Se applicabile.
```

---

## 💡 Proporre Features

### Prima di Proporre

1. **Cerca nelle Issues** - Potrebbe essere già stata proposta
2. **Considera la portata** - Feature complesse richiedono discussione
3. **Pensa all'impatto** - Come influenza gli utenti esistenti?

### Template Issue Feature

```markdown
## Problema/Motivazione

Descrivi il problema che questa feature risolve.

## Soluzione Proposta

Descrivi come vorresti che funzionasse.

## Alternative Considerate

Altre soluzioni che hai considerato.

## Impatto

- [ ] Breaking change
- [ ] Nuove dipendenze
- [ ] Cambiamenti al config

## Implementazione

Sei disposto a implementarla? Hai bisogno di aiuto?
```

---

## 🙏 Ringraziamenti

Ogni contributo, grande o piccolo, è apprezzato. Grazie per rendere questo progetto migliore!

---

*Per domande, apri una Issue o contatta [il maintainer].*


"""
Memoria Procedurale tramite SOP (Standard Operating Procedures).

Questo modulo implementa un sistema di memoria procedurale che permette
al modello di seguire procedure strutturate passo-passo.

Le SOP sono composte da:
- Nome e descrizione
- Trigger (quando attivare la procedura)
- Step sequenziali con condizioni
- Azioni da eseguire
- Validazione dei risultati

Uso:
    ```python
    from src.memory.procedural_memory import SOPManager, SOP, SOPStep
    
    # Crea una SOP
    sop = SOP(
        name="debug_code",
        description="Procedura per debuggare codice Python",
        trigger="utente chiede di debuggare o trovare bug",
        steps=[
            SOPStep(action="Leggi il codice e identifica il problema"),
            SOPStep(action="Proponi una soluzione", condition="problema identificato"),
            SOPStep(action="Verifica la soluzione"),
        ]
    )
    
    # Usa il manager
    manager = SOPManager()
    manager.add_sop(sop)
    
    # Trova SOP rilevante
    relevant = manager.find_relevant_sop("come faccio a debuggare questo codice?")
    ```
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Stato di esecuzione di uno step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class SOPStep:
    """
    Singolo step di una SOP.
    
    Attributes:
        action: Descrizione dell'azione da eseguire
        condition: Condizione per eseguire lo step (None = sempre)
        expected_output: Output atteso (per validazione)
        tools: Tool da usare in questo step
        fallback: Azione alternativa se lo step fallisce
    """
    action: str
    condition: Optional[str] = None
    expected_output: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    fallback: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte lo step in dizionario."""
        return {
            "action": self.action,
            "condition": self.condition,
            "expected_output": self.expected_output,
            "tools": self.tools,
            "fallback": self.fallback,
            "status": self.status.value,
            "result": self.result,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SOPStep":
        """Crea uno step da dizionario."""
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = StepStatus(status)
        
        return cls(
            action=data["action"],
            condition=data.get("condition"),
            expected_output=data.get("expected_output"),
            tools=data.get("tools", []),
            fallback=data.get("fallback"),
            status=status,
            result=data.get("result"),
        )


@dataclass
class SOP:
    """
    Standard Operating Procedure (SOP).
    
    Una procedura strutturata che guida il modello attraverso
    una serie di step per completare un task complesso.
    
    Attributes:
        name: Nome univoco della SOP
        description: Descrizione della procedura
        trigger: Pattern/keywords che attivano questa SOP
        category: Categoria (coding, debugging, documentation, etc.)
        steps: Lista di step da eseguire
        priority: Priorità (1-10, più alto = più importante)
        enabled: Se la SOP è attiva
    """
    name: str
    description: str
    trigger: str
    steps: List[SOPStep]
    category: str = "general"
    priority: int = 5
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte la SOP in dizionario."""
        return {
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "category": self.category,
            "steps": [s.to_dict() for s in self.steps],
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SOP":
        """Crea una SOP da dizionario."""
        steps = [SOPStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            name=data["name"],
            description=data["description"],
            trigger=data["trigger"],
            category=data.get("category", "general"),
            steps=steps,
            priority=data.get("priority", 5),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )
    
    def to_prompt(self) -> str:
        """
        Genera una rappresentazione della SOP per il prompt.
        
        Returns:
            Stringa formattata per includere nel contesto del modello
        """
        lines = [
            f"## Procedura: {self.name}",
            f"**Descrizione**: {self.description}",
            f"**Categoria**: {self.category}",
            "",
            "### Steps:",
        ]
        
        for i, step in enumerate(self.steps, 1):
            step_line = f"{i}. {step.action}"
            if step.condition:
                step_line += f" (se: {step.condition})"
            if step.tools:
                step_line += f" [tools: {', '.join(step.tools)}]"
            lines.append(step_line)
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Resetta tutti gli step a pending."""
        for step in self.steps:
            step.status = StepStatus.PENDING
            step.result = None


# =============================================================================
# SOP MANAGER
# =============================================================================

class SOPManager:
    """
    Gestore delle SOP (Standard Operating Procedures).
    
    Permette di:
    - Caricare/salvare SOP da file
    - Trovare SOP rilevanti per una query
    - Eseguire SOP passo-passo
    - Generare contesto per il modello
    """
    
    def __init__(self, sop_directory: Optional[str] = None):
        """
        Inizializza il manager.
        
        Args:
            sop_directory: Directory per caricare/salvare SOP (None = solo in memoria)
        """
        self.sops: Dict[str, SOP] = {}
        self.sop_directory = sop_directory
        
        # Carica SOP da directory se specificata
        if sop_directory and os.path.exists(sop_directory):
            self.load_sops_from_directory(sop_directory)
        
        # Aggiungi SOP di default
        self._add_default_sops()
    
    def _add_default_sops(self) -> None:
        """Aggiunge SOP di default per task comuni."""
        
        # SOP: Debug codice
        self.add_sop(SOP(
            name="debug_python_code",
            description="Procedura per identificare e risolvere bug nel codice Python",
            trigger="debug, bug, errore, non funziona, problema nel codice, fix",
            category="coding",
            priority=8,
            steps=[
                SOPStep(
                    action="Leggi attentamente il codice e l'errore riportato",
                    expected_output="Comprensione del contesto",
                ),
                SOPStep(
                    action="Identifica il tipo di errore (sintassi, logica, runtime)",
                    expected_output="Classificazione dell'errore",
                ),
                SOPStep(
                    action="Localizza la riga o la funzione problematica",
                    expected_output="Posizione del bug",
                ),
                SOPStep(
                    action="Proponi una soluzione con spiegazione",
                    expected_output="Codice corretto + spiegazione",
                ),
                SOPStep(
                    action="Suggerisci test per verificare il fix",
                    expected_output="Casi di test",
                ),
            ],
        ))
        
        # SOP: Scrittura codice
        self.add_sop(SOP(
            name="write_python_function",
            description="Procedura per scrivere una funzione Python di qualità",
            trigger="scrivi, crea, implementa, funzione, codice",
            category="coding",
            priority=7,
            steps=[
                SOPStep(
                    action="Comprendi i requisiti e i casi d'uso",
                    expected_output="Lista requisiti",
                ),
                SOPStep(
                    action="Definisci la firma della funzione con type hints",
                    expected_output="def function_name(params) -> ReturnType",
                ),
                SOPStep(
                    action="Scrivi la docstring con descrizione, args, returns",
                    expected_output="Docstring completa",
                ),
                SOPStep(
                    action="Implementa la logica con gestione errori",
                    expected_output="Codice funzionante",
                ),
                SOPStep(
                    action="Aggiungi validazione input se necessario",
                    condition="input complessi o da utente",
                ),
                SOPStep(
                    action="Fornisci esempio d'uso",
                    expected_output="Esempio chiamata",
                ),
            ],
        ))
        
        # SOP: Ricerca RAG
        self.add_sop(SOP(
            name="rag_search_procedure",
            description="Procedura per rispondere usando la knowledge base",
            trigger="cerca, trova, knowledge base, documentazione, RAG",
            category="rag",
            priority=9,
            steps=[
                SOPStep(
                    action="Identifica le parole chiave dalla domanda",
                    expected_output="Keywords per la ricerca",
                ),
                SOPStep(
                    action="Cerca nella knowledge base usando search_knowledge_base",
                    tools=["search_knowledge_base"],
                    expected_output="Documenti rilevanti",
                ),
                SOPStep(
                    action="Valuta la rilevanza dei risultati",
                    expected_output="Risultati filtrati",
                ),
                SOPStep(
                    action="Sintetizza la risposta citando le fonti",
                    expected_output="Risposta con citazioni",
                ),
                SOPStep(
                    action="Se non trovi informazioni, ammettilo chiaramente",
                    condition="nessun risultato rilevante",
                    expected_output="Ammissione mancanza info",
                ),
            ],
        ))
        
        # SOP: Code Review
        self.add_sop(SOP(
            name="code_review",
            description="Procedura per fare code review",
            trigger="review, revisiona, controlla il codice, feedback",
            category="coding",
            priority=7,
            steps=[
                SOPStep(
                    action="Verifica la correttezza logica",
                    expected_output="Lista issue logici",
                ),
                SOPStep(
                    action="Controlla lo stile e le convenzioni (PEP8)",
                    expected_output="Issue di stile",
                ),
                SOPStep(
                    action="Valuta la gestione degli errori",
                    expected_output="Copertura errori",
                ),
                SOPStep(
                    action="Verifica type hints e documentazione",
                    expected_output="Completezza docs",
                ),
                SOPStep(
                    action="Suggerisci ottimizzazioni se appropriate",
                    condition="codice funzionante ma migliorabile",
                ),
                SOPStep(
                    action="Fornisci feedback costruttivo",
                    expected_output="Riepilogo review",
                ),
            ],
        ))
        
        # SOP: Spiegazione concetto
        self.add_sop(SOP(
            name="explain_concept",
            description="Procedura per spiegare un concetto tecnico",
            trigger="spiega, cos'è, come funziona, cosa significa",
            category="education",
            priority=6,
            steps=[
                SOPStep(
                    action="Fornisci una definizione semplice (1-2 frasi)",
                    expected_output="Definizione base",
                ),
                SOPStep(
                    action="Spiega il concetto in dettaglio",
                    expected_output="Spiegazione approfondita",
                ),
                SOPStep(
                    action="Fornisci un'analogia o esempio pratico",
                    expected_output="Esempio concreto",
                ),
                SOPStep(
                    action="Mostra un esempio di codice se appropriato",
                    condition="concetto programmazione",
                    expected_output="Codice esempio",
                ),
                SOPStep(
                    action="Indica risorse per approfondire",
                    expected_output="Link/riferimenti",
                ),
            ],
        ))
    
    def add_sop(self, sop: SOP) -> None:
        """Aggiunge una SOP al manager."""
        self.sops[sop.name] = sop
        logger.debug(f"SOP aggiunta: {sop.name}")
    
    def remove_sop(self, name: str) -> bool:
        """Rimuove una SOP."""
        if name in self.sops:
            del self.sops[name]
            return True
        return False
    
    def get_sop(self, name: str) -> Optional[SOP]:
        """Ottiene una SOP per nome."""
        return self.sops.get(name)
    
    def list_sops(self, category: Optional[str] = None) -> List[SOP]:
        """
        Lista tutte le SOP.
        
        Args:
            category: Filtra per categoria (None = tutte)
            
        Returns:
            Lista di SOP ordinate per priorità
        """
        sops = list(self.sops.values())
        
        if category:
            sops = [s for s in sops if s.category == category]
        
        return sorted(sops, key=lambda x: -x.priority)
    
    def find_relevant_sop(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 1,
    ) -> List[SOP]:
        """
        Trova le SOP più rilevanti per una query.
        
        Usa matching semplice sui trigger. Per matching semantico,
        integrare con il VectorStore.
        
        Args:
            query: Query dell'utente
            category: Filtro categoria
            top_k: Numero di SOP da restituire
            
        Returns:
            Lista delle SOP più rilevanti
        """
        query_lower = query.lower()
        scored_sops = []
        
        for sop in self.sops.values():
            if not sop.enabled:
                continue
            
            if category and sop.category != category:
                continue
            
            # Calcola score basato su trigger match
            trigger_words = sop.trigger.lower().split(",")
            score = 0
            
            for trigger in trigger_words:
                trigger = trigger.strip()
                if trigger in query_lower:
                    score += 10
                # Match parziale
                for word in trigger.split():
                    if word in query_lower:
                        score += 2
            
            # Boost per priorità
            score += sop.priority * 0.5
            
            if score > 0:
                scored_sops.append((sop, score))
        
        # Ordina per score
        scored_sops.sort(key=lambda x: -x[1])
        
        return [sop for sop, _ in scored_sops[:top_k]]
    
    def get_sop_context(self, query: str) -> str:
        """
        Genera il contesto SOP per il prompt del modello.
        
        Args:
            query: Query dell'utente
            
        Returns:
            Stringa con la SOP da seguire (o vuota se nessuna)
        """
        relevant = self.find_relevant_sop(query, top_k=1)
        
        if not relevant:
            return ""
        
        sop = relevant[0]
        
        context = [
            "---",
            "**PROCEDURA DA SEGUIRE:**",
            "",
            sop.to_prompt(),
            "",
            "Segui questa procedura passo-passo. Indica quale step stai eseguendo.",
            "---",
        ]
        
        return "\n".join(context)
    
    def save_sop(self, sop: SOP, filepath: Optional[str] = None) -> str:
        """
        Salva una SOP su file JSON.
        
        Args:
            sop: SOP da salvare
            filepath: Path del file (None = usa sop_directory)
            
        Returns:
            Path del file salvato
        """
        if filepath is None:
            if self.sop_directory is None:
                raise ValueError("Nessuna directory SOP configurata")
            os.makedirs(self.sop_directory, exist_ok=True)
            filepath = os.path.join(self.sop_directory, f"{sop.name}.json")
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sop.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"SOP salvata: {filepath}")
        return filepath
    
    def load_sop(self, filepath: str) -> SOP:
        """Carica una SOP da file JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        sop = SOP.from_dict(data)
        self.add_sop(sop)
        return sop
    
    def load_sops_from_directory(self, directory: str) -> int:
        """
        Carica tutte le SOP da una directory.
        
        Args:
            directory: Directory con file .json
            
        Returns:
            Numero di SOP caricate
        """
        count = 0
        path = Path(directory)
        
        for json_file in path.glob("*.json"):
            try:
                self.load_sop(str(json_file))
                count += 1
            except Exception as e:
                logger.warning(f"Errore caricamento {json_file}: {e}")
        
        logger.info(f"Caricate {count} SOP da {directory}")
        return count
    
    def export_all(self, directory: str) -> int:
        """
        Esporta tutte le SOP in una directory.
        
        Args:
            directory: Directory di destinazione
            
        Returns:
            Numero di SOP esportate
        """
        os.makedirs(directory, exist_ok=True)
        count = 0
        
        for sop in self.sops.values():
            filepath = os.path.join(directory, f"{sop.name}.json")
            self.save_sop(sop, filepath)
            count += 1
        
        return count


# =============================================================================
# SYSTEM PROMPT CON SOP
# =============================================================================

SYSTEM_PROMPT_WITH_SOP = """You are an AI assistant that follows Standard Operating Procedures (SOPs) when applicable.

When you identify that a task matches a known procedure:
1. State which procedure you're following
2. Execute each step in order
3. Report the result of each step
4. Skip steps with unmet conditions
5. Provide a summary at the end

{sop_context}

If no procedure is applicable, respond naturally and helpfully."""


def get_system_prompt_with_sop(query: str, sop_manager: SOPManager) -> str:
    """
    Genera il system prompt con la SOP appropriata.
    
    Args:
        query: Query dell'utente
        sop_manager: Manager delle SOP
        
    Returns:
        System prompt completo
    """
    sop_context = sop_manager.get_sop_context(query)
    return SYSTEM_PROMPT_WITH_SOP.format(sop_context=sop_context)


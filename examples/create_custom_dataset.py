"""
Script per creare dataset personalizzati per instruction tuning.

Questo script mostra come creare dataset in vari formati (JSON, JSONL, CSV)
per il fine-tuning di LLM.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any


def create_jsonl_dataset(
    examples: List[Dict[str, str]],
    output_path: str = "data/custom_dataset.jsonl",
) -> None:
    """
    Crea un dataset in formato JSONL (una riga = un esempio JSON).
    
    Formato richiesto:
    {
        "instruction": "...",
        "response": "..."
    }
    
    Args:
        examples: Lista di dizionari con 'instruction' e 'response'
        output_path: Path dove salvare il file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Dataset JSONL creato: {output_path} ({len(examples)} esempi)")


def create_json_dataset(
    examples: List[Dict[str, str]],
    output_path: str = "data/custom_dataset.json",
) -> None:
    """
    Crea un dataset in formato JSON (array di oggetti).
    
    Args:
        examples: Lista di dizionari con 'instruction' e 'response'
        output_path: Path dove salvare il file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset JSON creato: {output_path} ({len(examples)} esempi)")


def create_csv_dataset(
    examples: List[Dict[str, str]],
    output_path: str = "data/custom_dataset.csv",
) -> None:
    """
    Crea un dataset in formato CSV.
    
    Args:
        examples: Lista di dizionari con 'instruction' e 'response'
        output_path: Path dove salvare il file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response"])
        writer.writeheader()
        writer.writerows(examples)
    
    print(f"Dataset CSV creato: {output_path} ({len(examples)} esempi)")


def example_italian_dataset() -> List[Dict[str, str]]:
    """
    Esempio di dataset in italiano.
    
    Returns:
        Lista di esempi instruction-response
    """
    return [
        {
            "instruction": "Spiega cos'è l'intelligenza artificiale in modo semplice.",
            "response": "L'intelligenza artificiale (IA) è la capacità di un computer o di un sistema di eseguire compiti che normalmente richiederebbero l'intelligenza umana, come il riconoscimento di immagini, la comprensione del linguaggio naturale e il processo decisionale."
        },
        {
            "instruction": "Quali sono i vantaggi dell'apprendimento automatico?",
            "response": "I vantaggi dell'apprendimento automatico includono: automazione di compiti ripetitivi, capacità di analizzare grandi quantità di dati, miglioramento continuo delle prestazioni con più dati, personalizzazione dei servizi e capacità di identificare pattern complessi che gli umani potrebbero non notare."
        },
        {
            "instruction": "Come funziona una rete neurale?",
            "response": "Una rete neurale è composta da strati di nodi (neuroni artificiali) collegati tra loro. Ogni connessione ha un peso che viene regolato durante l'addestramento. I dati passano attraverso la rete, e ogni strato elabora le informazioni prima di passarle al successivo. Durante l'addestramento, la rete modifica i pesi per minimizzare l'errore tra le previsioni e i risultati attesi."
        },
        {
            "instruction": "Qual è la differenza tra deep learning e machine learning tradizionale?",
            "response": "Il machine learning tradizionale richiede spesso l'estrazione manuale di feature dai dati, mentre il deep learning utilizza reti neurali profonde che apprendono automaticamente le feature dai dati grezzi. Il deep learning è particolarmente efficace per dati non strutturati come immagini, testo e audio, e può gestire problemi più complessi grazie alla sua capacità di apprendere rappresentazioni gerarchiche."
        },
        {
            "instruction": "Spiega il concetto di overfitting nel machine learning.",
            "response": "L'overfitting si verifica quando un modello apprende troppo bene i dati di addestramento, memorizzando i dettagli specifici invece di apprendere pattern generali. Questo porta a prestazioni eccellenti sui dati di addestramento ma scarse prestazioni su dati nuovi. Per prevenirlo, si possono usare tecniche come la regolarizzazione, il dropout, la validazione incrociata e l'arresto anticipato."
        },
    ]


def example_qa_dataset() -> List[Dict[str, str]]:
    """
    Esempio di dataset domanda-risposta.
    
    Returns:
        Lista di esempi Q&A
    """
    return [
        {
            "instruction": "Qual è la capitale della Francia?",
            "response": "La capitale della Francia è Parigi."
        },
        {
            "instruction": "Chi ha scritto 'Il Principe'?",
            "response": "'Il Principe' è stato scritto da Niccolò Machiavelli nel 1513."
        },
        {
            "instruction": "Quanto dura un anno su Marte?",
            "response": "Un anno su Marte dura circa 687 giorni terrestri, equivalente a circa 1,88 anni terrestri."
        },
    ]


def example_code_dataset() -> List[Dict[str, str]]:
    """
    Esempio di dataset per code generation/explanation.
    
    Returns:
        Lista di esempi code-related
    """
    return [
        {
            "instruction": "Scrivi una funzione Python che calcola il fattoriale di un numero.",
            "response": "```python\ndef fattoriale(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * fattoriale(n - 1)\n\n# Esempio d'uso:\nprint(fattoriale(5))  # Output: 120\n```"
        },
        {
            "instruction": "Spiega cosa fa questo codice Python: `[x**2 for x in range(10)]`",
            "response": "Questo codice crea una list comprehension che genera una lista dei quadrati dei numeri da 0 a 9. Specificamente:\n- `range(10)` genera i numeri da 0 a 9\n- `x**2` calcola il quadrato di ogni numero\n- Il risultato è `[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]`"
        },
    ]


def main():
    """Crea esempi di dataset personalizzati."""
    # Crea directory per i dataset
    Path("data").mkdir(exist_ok=True)
    
    # Dataset italiano
    italian_data = example_italian_dataset()
    create_jsonl_dataset(italian_data, "data/italian_ai_dataset.jsonl")
    create_json_dataset(italian_data, "data/italian_ai_dataset.json")
    create_csv_dataset(italian_data, "data/italian_ai_dataset.csv")
    
    # Dataset Q&A
    qa_data = example_qa_dataset()
    create_jsonl_dataset(qa_data, "data/qa_dataset.jsonl")
    
    # Dataset code
    code_data = example_code_dataset()
    create_jsonl_dataset(code_data, "data/code_dataset.jsonl")
    
    print("\n✅ Dataset creati con successo!")
    print("\nPer usare questi dataset, modifica config/config.yaml:")
    print("  data:")
    print("    dataset_name: 'data/italian_ai_dataset.jsonl'")
    print("    text_column: 'instruction'")
    print("    response_column: 'response'")


if __name__ == "__main__":
    main()


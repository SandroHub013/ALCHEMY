"""
Esempio di uso della Memoria Procedurale (SOP).

Questo esempio mostra come:
1. Creare e gestire SOP
2. Trovare SOP rilevanti per una query
3. Generare contesto per il modello
4. Caricare SOP da file

Uso:
    python examples/sop_example.py
"""

from src.memory import SOPManager, SOP, SOPStep, get_system_prompt_with_sop


def main():
    print("=" * 60)
    print("ESEMPIO MEMORIA PROCEDURALE (SOP)")
    print("=" * 60)
    
    # 1. Crea manager con SOP di default
    print("\n[1] Creazione SOPManager...")
    manager = SOPManager(sop_directory="./data/sops")
    
    print(f"   SOP caricate: {len(manager.sops)}")
    
    # 2. Lista SOP disponibili
    print("\n[2] SOP disponibili:")
    for sop in manager.list_sops():
        print(f"   - {sop.name} ({sop.category}) - Priorità: {sop.priority}")
        print(f"     Trigger: {sop.trigger[:50]}...")
    
    # 3. Trova SOP per query
    print("\n[3] Test ricerca SOP:")
    test_queries = [
        "Ho un bug nel codice, come lo risolvo?",
        "Scrivi una funzione per calcolare fibonacci",
        "Cerca nella documentazione come usare RAG",
        "Cos'è il machine learning?",
    ]
    
    for query in test_queries:
        relevant = manager.find_relevant_sop(query, top_k=1)
        if relevant:
            sop = relevant[0]
            print(f"\n   Query: '{query[:40]}...'")
            print(f"   → SOP: {sop.name}")
        else:
            print(f"\n   Query: '{query[:40]}...'")
            print(f"   → Nessuna SOP trovata")
    
    # 4. Mostra procedura completa
    print("\n[4] Esempio procedura completa:")
    debug_sop = manager.get_sop("debug_python_code")
    if debug_sop:
        print(debug_sop.to_prompt())
    
    # 5. Genera system prompt
    print("\n[5] System prompt con SOP:")
    query = "Come faccio a debuggare questo codice Python?"
    prompt = get_system_prompt_with_sop(query, manager)
    print(prompt[:500] + "...")
    
    # 6. Crea SOP personalizzata
    print("\n[6] Creazione SOP personalizzata:")
    custom_sop = SOP(
        name="deploy_app",
        description="Procedura per deploy di applicazione",
        trigger="deploy, pubblica, metti in produzione",
        category="devops",
        priority=8,
        steps=[
            SOPStep(action="Verifica che tutti i test passino"),
            SOPStep(action="Crea build di produzione"),
            SOPStep(action="Backup del sistema attuale", condition="ambiente produzione"),
            SOPStep(action="Esegui il deploy"),
            SOPStep(action="Verifica che l'app funzioni correttamente"),
            SOPStep(action="Monitora per errori nei primi 30 minuti"),
        ]
    )
    manager.add_sop(custom_sop)
    print(f"   Aggiunta: {custom_sop.name}")
    
    # 7. Esporta SOP
    print("\n[7] Esportazione SOP...")
    manager.export_all("./data/sops")
    print(f"   SOP esportate in: ./data/sops")
    
    # 8. Statistiche
    print("\n[8] Statistiche:")
    categories = {}
    for sop in manager.sops.values():
        cat = sop.category
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"   Totale SOP: {len(manager.sops)}")
    print("   Per categoria:")
    for cat, count in sorted(categories.items()):
        print(f"     - {cat}: {count}")
    
    print("\n" + "=" * 60)
    print("✅ Demo SOP completata!")
    print("=" * 60)


if __name__ == "__main__":
    main()


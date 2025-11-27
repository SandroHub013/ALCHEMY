"""
Example of Procedural Memory (SOP) usage.

This example shows how to:
1. Create and manage SOPs
2. Find relevant SOPs for a query
3. Generate context for the model
4. Load SOPs from files

Usage:
    python examples/sop_example.py
"""

from src.memory import SOPManager, SOP, SOPStep, get_system_prompt_with_sop


def main():
    print("=" * 60)
    print("PROCEDURAL MEMORY (SOP) EXAMPLE")
    print("=" * 60)
    
    # 1. Create manager with default SOPs
    print("\n[1] Creating SOPManager...")
    manager = SOPManager(sop_directory="./data/sops")
    
    print(f"   SOPs loaded: {len(manager.sops)}")
    
    # 2. List available SOPs
    print("\n[2] Available SOPs:")
    for sop in manager.list_sops():
        print(f"   - {sop.name} ({sop.category}) - Priority: {sop.priority}")
        print(f"     Trigger: {sop.trigger[:50]}...")
    
    # 3. Find SOP for query
    print("\n[3] SOP search test:")
    test_queries = [
        "I have a bug in the code, how do I fix it?",
        "Write a function to calculate fibonacci",
        "Search the documentation on how to use RAG",
        "What is machine learning?",
    ]
    
    for query in test_queries:
        relevant = manager.find_relevant_sop(query, top_k=1)
        if relevant:
            sop = relevant[0]
            print(f"\n   Query: '{query[:40]}...'")
            print(f"   → SOP: {sop.name}")
        else:
            print(f"\n   Query: '{query[:40]}...'")
            print(f"   → No SOP found")
    
    # 4. Show complete procedure
    print("\n[4] Complete procedure example:")
    debug_sop = manager.get_sop("debug_python_code")
    if debug_sop:
        print(debug_sop.to_prompt())
    
    # 5. Generate system prompt
    print("\n[5] System prompt with SOP:")
    query = "How do I debug this Python code?"
    prompt = get_system_prompt_with_sop(query, manager)
    print(prompt[:500] + "...")
    
    # 6. Create custom SOP
    print("\n[6] Creating custom SOP:")
    custom_sop = SOP(
        name="deploy_app",
        description="Procedure for application deployment",
        trigger="deploy, publish, put in production",
        category="devops",
        priority=8,
        steps=[
            SOPStep(action="Verify that all tests pass"),
            SOPStep(action="Create production build"),
            SOPStep(action="Backup current system", condition="production environment"),
            SOPStep(action="Execute deployment"),
            SOPStep(action="Verify that the app works correctly"),
            SOPStep(action="Monitor for errors in the first 30 minutes"),
        ]
    )
    manager.add_sop(custom_sop)
    print(f"   Added: {custom_sop.name}")
    
    # 7. Export SOPs
    print("\n[7] Exporting SOPs...")
    manager.export_all("./data/sops")
    print(f"   SOPs exported to: ./data/sops")
    
    # 8. Statistics
    print("\n[8] Statistics:")
    categories = {}
    for sop in manager.sops.values():
        cat = sop.category
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"   Total SOPs: {len(manager.sops)}")
    print("   By category:")
    for cat, count in sorted(categories.items()):
        print(f"     - {cat}: {count}")
    
    print("\n" + "=" * 60)
    print("✅ SOP Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

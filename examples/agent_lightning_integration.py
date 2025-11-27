"""
Esempio avanzato di integrazione con Microsoft Agent Lightning.

Questo esempio mostra come usare Agent Lightning per:
1. Training RL con GRPO (Group Relative Policy Optimization)
2. Reward functions personalizzate
3. Tracciamento span per debugging
4. Generazione con valutazione

Per uso semplice, usa direttamente:
    python main_agent_lightning.py --config config/config.yaml

GitHub Agent Lightning: https://github.com/microsoft/agent-lightning
"""

from typing import Dict, Any, Optional
import yaml
import torch

from src.models import ModelLoader
from src.data import create_data_module
from src.agent import (
    AgentLightningTrainer,
    AgentLightningConfig,
    TrainingAlgorithm,
    RewardFunction,
    check_agent_lightning_available,
)


def custom_reward_function(
    prompt: str,
    generation: str,
    reference: Optional[str] = None,
) -> float:
    """
    Esempio di reward function personalizzata.
    
    Puoi creare la tua reward function per task specifici.
    Deve ritornare un float tra -1.0 e 1.0.
    
    Args:
        prompt: Il prompt originale
        generation: La risposta generata dal modello
        reference: Risposta di riferimento (opzionale)
        
    Returns:
        Reward score
    """
    reward = 0.0
    
    # Esempio: premia risposte che includono codice Python
    if "def " in generation or "class " in generation:
        reward += 0.3
    
    # Esempio: premia risposte in italiano se il prompt è in italiano
    italian_words = ["il", "la", "che", "di", "un", "per", "sono", "è"]
    if any(word in prompt.lower() for word in italian_words):
        # Il prompt è in italiano, premia risposte in italiano
        if any(word in generation.lower() for word in italian_words):
            reward += 0.2
    
    # Esempio: penalizza risposte troppo corte
    if len(generation) < 50:
        reward -= 0.3
    
    # Esempio: penalizza risposte ripetitive
    words = generation.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:
            reward -= 0.3  # Molto ripetitivo
    
    return max(-1.0, min(1.0, reward))


def main():
    """Esempio di uso avanzato con Agent Lightning."""
    
    # Verifica disponibilità
    if not check_agent_lightning_available():
        print("❌ Agent Lightning non installato!")
        print("   Installa con: pip install agentlightning")
        return
    
    print("✅ Agent Lightning disponibile")
    
    # Carica configurazione
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Carica modello
    print("🔄 Caricamento modello...")
    model_config = config["model"]
    peft_config = config["peft"]
    
    quantization_config = peft_config.get("quantization", {})
    if isinstance(quantization_config.get("bnb_4bit_compute_dtype"), str):
        dtype_str = quantization_config["bnb_4bit_compute_dtype"]
        quantization_config["bnb_4bit_compute_dtype"] = getattr(torch, dtype_str, torch.float16)
    
    model_loader = ModelLoader(
        model_name_or_path=model_config["name_or_path"],
        quantization_config=quantization_config,
        lora_config=peft_config.get("lora", {}),
    )
    
    model, tokenizer = model_loader.get_model_and_tokenizer()
    print(f"✅ Modello caricato: {model_config['name_or_path']}")
    
    # Crea configurazione Agent Lightning
    agl_config = AgentLightningConfig(
        algorithm=TrainingAlgorithm.GRPO,  # Usa RL!
        grpo_config={
            "num_generations": 4,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 512,
            "kl_coef": 0.1,
        },
        enable_tracing=True,
        reward_function="combined",  # O usa custom_reward_function
    )
    
    # Crea trainer con reward function custom
    trainer = AgentLightningTrainer(
        model=model,
        tokenizer=tokenizer,
        config=agl_config,
        reward_fn=custom_reward_function,  # Usa la nostra funzione custom!
    )
    
    print("✅ Trainer creato con reward function personalizzata")
    
    # Prepara dataset
    print("🔄 Preparazione dataset...")
    data_module = create_data_module(tokenizer=tokenizer, config=config)
    data_module.setup("fit")
    
    train_dataset = data_module.train_dataset
    eval_dataset = data_module.val_dataset
    
    print(f"✅ Dataset: {len(train_dataset)} training samples")
    
    # Training
    print("\n" + "=" * 60)
    print("🚀 AVVIO TRAINING RL CON GRPO")
    print("=" * 60)
    
    results = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=1,  # Esempio veloce
        batch_size=2,
        learning_rate=2e-5,
        output_dir="./checkpoints/agent_lightning_example",
    )
    
    print("\n✅ Training completato!")
    
    # Test generazione e valutazione reward
    print("\n" + "=" * 60)
    print("🧪 TEST GENERAZIONE E REWARD")
    print("=" * 60)
    
    test_prompts = [
        "Write a Python function to reverse a string.",
        "Spiega cosa sono le reti neurali.",
        "Call the weather API to get temperature in Rome.",
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        # Genera risposta
        response = trainer.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7,
        )
        print(f"🤖 Response: {response[:200]}...")
        
        # Valuta reward
        reward = trainer.evaluate_reward(prompt, response)
        print(f"⭐ Reward: {reward:.3f}")
        
        # Confronta con reward function built-in
        builtin_reward = RewardFunction.combined_reward(prompt, response)
        print(f"📊 Reward (built-in): {builtin_reward:.3f}")
    
    print("\n🎉 Esempio completato!")


if __name__ == "__main__":
    main()

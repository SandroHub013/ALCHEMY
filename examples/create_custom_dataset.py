"""
Script to create custom datasets for instruction tuning.

This script shows how to create datasets in various formats (JSON, JSONL, CSV)
for LLM fine-tuning.
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
    Create a dataset in JSONL format (one line = one JSON example).
    
    Required format:
    {
        "instruction": "...",
        "response": "..."
    }
    
    Args:
        examples: List of dictionaries with 'instruction' and 'response'
        output_path: Path where to save the file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"JSONL dataset created: {output_path} ({len(examples)} examples)")


def create_json_dataset(
    examples: List[Dict[str, str]],
    output_path: str = "data/custom_dataset.json",
) -> None:
    """
    Create a dataset in JSON format (array of objects).
    
    Args:
        examples: List of dictionaries with 'instruction' and 'response'
        output_path: Path where to save the file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"JSON dataset created: {output_path} ({len(examples)} examples)")


def create_csv_dataset(
    examples: List[Dict[str, str]],
    output_path: str = "data/custom_dataset.csv",
) -> None:
    """
    Create a dataset in CSV format.
    
    Args:
        examples: List of dictionaries with 'instruction' and 'response'
        output_path: Path where to save the file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response"])
        writer.writeheader()
        writer.writerows(examples)
    
    print(f"CSV dataset created: {output_path} ({len(examples)} examples)")


def example_ai_dataset() -> List[Dict[str, str]]:
    """
    Example AI/ML dataset.
    
    Returns:
        List of instruction-response examples
    """
    return [
        {
            "instruction": "Explain what artificial intelligence is in simple terms.",
            "response": "Artificial Intelligence (AI) is the ability of a computer or system to perform tasks that would normally require human intelligence, such as image recognition, natural language understanding, and decision-making."
        },
        {
            "instruction": "What are the advantages of machine learning?",
            "response": "The advantages of machine learning include: automation of repetitive tasks, ability to analyze large amounts of data, continuous performance improvement with more data, service personalization, and ability to identify complex patterns that humans might not notice."
        },
        {
            "instruction": "How does a neural network work?",
            "response": "A neural network is composed of layers of nodes (artificial neurons) connected to each other. Each connection has a weight that is adjusted during training. Data passes through the network, and each layer processes the information before passing it to the next. During training, the network modifies the weights to minimize the error between predictions and expected results."
        },
        {
            "instruction": "What is the difference between deep learning and traditional machine learning?",
            "response": "Traditional machine learning often requires manual feature extraction from data, while deep learning uses deep neural networks that automatically learn features from raw data. Deep learning is particularly effective for unstructured data like images, text, and audio, and can handle more complex problems due to its ability to learn hierarchical representations."
        },
        {
            "instruction": "Explain the concept of overfitting in machine learning.",
            "response": "Overfitting occurs when a model learns the training data too well, memorizing specific details instead of learning general patterns. This leads to excellent performance on training data but poor performance on new data. To prevent it, techniques like regularization, dropout, cross-validation, and early stopping can be used."
        },
    ]


def example_qa_dataset() -> List[Dict[str, str]]:
    """
    Example question-answer dataset.
    
    Returns:
        List of Q&A examples
    """
    return [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "instruction": "Who wrote 'The Prince'?",
            "response": "'The Prince' was written by Niccolò Machiavelli in 1513."
        },
        {
            "instruction": "How long is a year on Mars?",
            "response": "A year on Mars lasts about 687 Earth days, equivalent to approximately 1.88 Earth years."
        },
    ]


def example_code_dataset() -> List[Dict[str, str]]:
    """
    Example dataset for code generation/explanation.
    
    Returns:
        List of code-related examples
    """
    return [
        {
            "instruction": "Write a Python function that calculates the factorial of a number.",
            "response": "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n\n# Usage example:\nprint(factorial(5))  # Output: 120\n```"
        },
        {
            "instruction": "Explain what this Python code does: `[x**2 for x in range(10)]`",
            "response": "This code creates a list comprehension that generates a list of squares of numbers from 0 to 9. Specifically:\n- `range(10)` generates numbers from 0 to 9\n- `x**2` calculates the square of each number\n- The result is `[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]`"
        },
    ]


def main():
    """Create custom dataset examples."""
    # Create directory for datasets
    Path("data").mkdir(exist_ok=True)
    
    # AI dataset
    ai_data = example_ai_dataset()
    create_jsonl_dataset(ai_data, "data/ai_dataset.jsonl")
    create_json_dataset(ai_data, "data/ai_dataset.json")
    create_csv_dataset(ai_data, "data/ai_dataset.csv")
    
    # Q&A dataset
    qa_data = example_qa_dataset()
    create_jsonl_dataset(qa_data, "data/qa_dataset.jsonl")
    
    # Code dataset
    code_data = example_code_dataset()
    create_jsonl_dataset(code_data, "data/code_dataset.jsonl")
    
    print("\n✅ Datasets created successfully!")
    print("\nTo use these datasets, modify config/config.yaml:")
    print("  data:")
    print("    dataset_name: 'data/ai_dataset.jsonl'")
    print("    text_column: 'instruction'")
    print("    response_column: 'response'")


if __name__ == "__main__":
    main()

from langsmith import wrappers, Client
from openai import AsyncOpenAI
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import asyncio

from api.services.classifier import FurnitureCategoryClassifier

# Load environment variables from .env file
load_dotenv()

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

DATASET_NAME = "CSV Dataset for Wabi Catalog"

# Check if dataset already exists
existing_datasets =  client.read_dataset(dataset_name=DATASET_NAME)

if existing_datasets:
    print(f"✅ Dataset '{DATASET_NAME}' already exists. Using existing dataset.")
    dataset = existing_datasets  # Reuse existing
else:
    print(f"📦 Uploading new dataset: {DATASET_NAME}")
    csv_file = 'data/Furniture_Catalog__English_Placement_.csv'
    input_keys = ['Product URL']
    output_keys = ['Category']

    dataset = client.upload_csv(
        csv_file=csv_file,
        input_keys=input_keys,
        output_keys=output_keys,
        name=DATASET_NAME,
        description="Dataset containing catalog information for Wabi Dashboard",
        data_type="kv"
    )


classifier = FurnitureCategoryClassifier()  # Create instance once globally

def target_function(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Target function to classify a single input using FurnitureCategoryClassifier.
    This wraps the async batch method to support synchronous LangSmith usage.
    """
    try:
        # Wrap input in a batch (as list of dicts)
        batch_input = [inputs]

        # Run classification (simulate batch_id=1 for now)
        batch_result = asyncio.run(classifier.classify_image_batch(batch_input, batch_id=1))

        # Return first result's category
        category = batch_result.results[0].category
        return {'Category': category}

    except Exception as e:
        print(f"Error in target_function: {e}")
        return {'Category': 'Error'}


def strict_rule_based_evaluator(run, example) -> Dict[str, Any]:
    """
    Strict rule-based heuristic evaluator that checks exact equality between input and output.

    Args:
        run: The run object containing the prediction results
        example: The example object containing the ground truth

    Returns:
        Dictionary containing evaluation results with binary pass/fail
    """
    # Extract predicted output from the run
    predicted_output = run.outputs if run.outputs else {}
    predicted_category = predicted_output.get('Category', '')

    # Extract expected output from the example
    expected_output = example.outputs if example.outputs else {}
    expected_category = expected_output.get('Category', '')

    # STRICT RULE: Exact string match (case-sensitive, no preprocessing)
    is_exact_match = predicted_category == expected_category

    # Binary scoring: 1 for pass, 0 for fail
    score = 1 if is_exact_match else 0

    evaluation_results = {
        'score': score,
        'key': 'strict_exact_match',
        'comment': f"STRICT: '{predicted_category}' == '{expected_category}' -> {is_exact_match}",
        'correct': is_exact_match
    }

    return evaluation_results

# Example usage:
if __name__ == "__main__":
    # Run evaluation on the dataset
    from langsmith.evaluation import evaluate

    # Evaluate using strict rule-based heuristic
    results = evaluate(
        target_function,
        data=dataset,
        evaluators=[strict_rule_based_evaluator],
        experiment_prefix="wabi-catalog-image-strict-evaluation"
    )

    print("Strict rule-based evaluation completed!")

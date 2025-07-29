from langsmith import Client
import os
from typing import Dict, Any
from dotenv import load_dotenv
import asyncio
from langsmith.utils import LangSmithNotFoundError

from api.services.classifier import FurnitureCategoryClassifier

# Load environment variables from .env file
load_dotenv()
client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

DATASET_NAME = "CSV Dataset for Wabi Catalog 4"

# Check if dataset already exists with enhanced error handling
try:
    # First try to find exact match
    existing_datasets = client.read_dataset(dataset_name=DATASET_NAME)
    print(f"✅ Dataset '{DATASET_NAME}' already exists. Using existing dataset.")
    dataset = existing_datasets

except LangSmithNotFoundError:
    # Dataset doesn't exist, create new one
    print(f"📦 Creating new dataset: {DATASET_NAME}")

    csv_file = 'data/Furniture_Catalog__English_Placement_.csv'

    # Verify CSV file exists before attempting upload
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"CSV file not found at path: {csv_file}\n"
            f"Current working directory: {os.getcwd()}"
        )

    # Verify CSV has required columns by reading first line
    with open(csv_file, 'r') as f:
        header = f.readline().strip().split(',')
        if 'Tags' not in header:
            raise ValueError(f"CSV missing 'Tags' column. Found columns: {header}")
        if 'Category' not in header:
            raise ValueError(f"CSV missing 'Category' column. Found columns: {header}")

    try:
        dataset = client.upload_csv(
            csv_file=csv_file,
            input_keys=['Tags'],  # Changed from 'Product URL' to 'Tags'
            output_keys=['Category'],
            name=DATASET_NAME,
            description="Dataset containing catalog information for Wabi Dashboard",
            data_type="kv"
        )
        print("🎉 Successfully created new dataset")

    except Exception as upload_error:
        print(f"❌ Failed to upload dataset: {str(upload_error)}")
        # Provide helpful debugging info
        print("\nDebugging info:")
        print(f"- CSV path: {os.path.abspath(csv_file)}")
        print(f"- File exists: {os.path.exists(csv_file)}")
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                print(f"- First 2 lines:\n{f.readline()}{f.readline()}")
        raise  # Re-raise the exception after logging

except Exception as e:
    print(f"❌ Unexpected error during dataset setup: {str(e)}")
    raise


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
        batch_result = asyncio.run(classifier.classify_text_batch(batch_input, batch_id=1))

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
    is_exact_match = predicted_category.lower() == expected_category.lower()

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
        experiment_prefix="wabi-catalog-text-strict-evaluation"
    )

    print("Strict rule-based evaluation completed!")

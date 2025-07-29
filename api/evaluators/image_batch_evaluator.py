from langsmith import Client
import os
from typing import Dict, Any
from dotenv import load_dotenv
from urllib.parse import urlparse
import logging
import asyncio
from api.services.image_classifier import ImageClassifier
from langsmith.utils import LangSmithNotFoundError
import re

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
            input_keys=['Product URL', 'Tags'],  # Changed from 'Product URL' to 'Tags'
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


classifier = ImageClassifier()  # Create instance once globally

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_product_url(url: str) -> str:
    """Enhanced URL validation specifically for furniture product URLs"""
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    url = url.strip()

    # Common fixes for product URLs
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'

    # Remove tracking parameters if they exist
    url = re.sub(r'\?piid=[0-9,%]+$', '', url)

    # Validate URL structure
    parsed = urlparse(url)
    if not all([parsed.scheme, parsed.netloc]):
        raise ValueError(f"Invalid URL structure: {url}")

    # Additional checks for known furniture domains
    allowed_domains = [
        'wayfair.com',
        'ikea.com',
        'amazon.com',
        'jossandmain.com',
        'allmodern.com'
    ]
    if not any(domain in parsed.netloc for domain in allowed_domains):
        logger.warning(f"URL from unexpected domain: {parsed.netloc}")

    return url

def target_function(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final optimized target function for furniture classification that:
    1. Handles multiple input formats from LangSmith
    2. Validates and cleans product URLs
    3. Processes through ImageClassifier
    4. Returns standardized Category output
    """
    # Debug: Log raw input structure
    logger.debug(f"Raw inputs received: {inputs}")

    try:
    #EXTRACT PRODUCT URL (multiple possible formats)
        product_url = None

        # Try all possible input paths
        possible_paths = [
            inputs.get('Product URL'),
            inputs.get('product_url'),
            inputs.get('input', {}).get('Product URL'),
            inputs.get('data', {}).get('Product URL'),
            inputs.get('Product_URL')  # Handle snake case
        ]

        for url in possible_paths:
            if url and isinstance(url, str) and url.strip():
                product_url = url.strip()
                break

        if not product_url:
            logger.error(f"No valid Product URL found in: {inputs.keys()}")
            return {"Category": "Error: No Product URL provided in input data"}

        #VALIDATE AND CLEAN PRODUCT URL
        try:
            clean_url = validate_product_url(product_url)
            logger.info(f"Processing product: {clean_url}")
        except ValueError as e:
            logger.error(f"URL validation failed for {product_url}: {str(e)}")
            return {"Category": f"Error: Invalid Product URL - {str(e)}"}

        # RUN CLASSIFICATION
        try:
            batch_result = asyncio.run(
                classifier.classify_image_batch(
                    [{"Product URL": clean_url}],
                    batch_id=1
                )
            )
            logger.debug(f"Classification result: {batch_result}")

            # PROCESS RESULTS
            if not batch_result.get("results"):
                logger.error("Empty results from classifier")
                return {"Category": "Error: No classification results returned"}

            first_result = batch_result["results"][0]

            if first_result.get("success"):
                category = first_result["result"].get("category")
                if not category:
                    logger.warning("Classification succeeded but no category returned")
                    return {"Category": "Error: No category in results"}

                logger.info(f"Classification success: {category}")
                return {"Category": category}

            else:
                error_msg = first_result.get("error", "Unknown classification error")
                logger.error(f"Classification failed: {error_msg}")
                return {"Category": f"Error: {error_msg}"}

        except Exception as e:
            logger.error(f"Classification processing error: {str(e)}", exc_info=True)
            return {"Category": f"Error: Classification processing failed - {str(e)}"}

    except Exception as e:
        logger.critical(f"Unexpected error in target_function: {str(e)}", exc_info=True)
        return {"Category": f"Error: System error - {str(e)}"}

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
        experiment_prefix="wabi-catalog-image-strict-evaluation"
    )

    print("Strict rule-based evaluation completed!")

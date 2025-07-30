import openai
import asyncio
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from config import settings
from api.models import CategoryResult, BatchResult
from api.utils.constants import CATEGORY_CLASSIFICATION_PROMPT, CATEGORY_CLASSIFICATION_PROMPT_TEXT_BASED


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextBasedFurnitureClassifier:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    async def classify_text_batch(self, batch_data: List[Dict[str, Any]], batch_id: int) -> BatchResult:
        """Classify furniture categories using text data"""
        try:
            logger.info(f"Processing text batch {batch_id} with {len(batch_data)} items")

            # Prepare furniture data for each item
            furniture_descriptions = []
            for item in batch_data:
                # Extract relevant text fields
                text_fields = []
                for key, value in item.items():
                    if key not in ["Product Name", "Tags"] and value:
                        text_fields.append(f"{key}: {value}")
                furniture_descriptions.append(" | ".join(text_fields))

            # Create prompt for batch
            batch_prompt = f"{CATEGORY_CLASSIFICATION_PROMPT_TEXT_BASED}\n\nFurniture items to classify:\n"
            for i, desc in enumerate(furniture_descriptions):
                batch_prompt += f"{i+1}. {desc}\n"

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CATEGORY_CLASSIFICATION_PROMPT_TEXT_BASED},
                    {"role": "user", "content": f"Classify these furniture items into categories:\n{batch_prompt}"}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            # Parse response
            classification_text = response.choices[0].message.content

            # Try to extract JSON from response
            results = []
            try:
                # Attempt to parse as JSON array
                if classification_text.strip().startswith('['):
                    parsed_results = json.loads(classification_text)
                else:
                    # Extract individual JSON objects
                    import re
                    json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', classification_text)
                    parsed_results = [json.loads(match) for match in json_matches]

                # Convert to structured results
                for i, result_data in enumerate(parsed_results):
                    if i < len(batch_data):
                        result = self._create_category_result(result_data, batch_data[i], i)
                        results.append(result)

                # Fill missing results if needed
                while len(results) < len(batch_data):
                    i = len(results)
                    fallback_result = self._create_fallback_result(batch_data[i], i)
                    results.append(fallback_result)

            except json.JSONDecodeError:
                # Fallback: create basic results
                results = [
                    self._create_fallback_result(item, i)
                    for i, item in enumerate(batch_data)
                ]

            return BatchResult(
                batch_id=batch_id,
                results=results,
                timestamp=datetime.now().isoformat(),
                processing_type="text"
            )

        except Exception as e:
            logger.error(f"Error in text classification: {str(e)}")
            return BatchResult(
                batch_id=batch_id,
                results=[
                    self._create_error_result(item, i, str(e))
                    for i, item in enumerate(batch_data)
                ],
                timestamp=datetime.now().isoformat(),
                processing_type="text"
            )

    def _create_category_result(self, result_data: Dict[str, Any], original_item: Dict[str, Any], index: int) -> CategoryResult:
        """Create category classification result from parsed data"""
        # Extract product name
        product_name = (
            result_data.get('product_name') or
            original_item.get('Product Name') or
            original_item.get('product_name') or
            f"Furniture Item {index + 1}"
        )

        return CategoryResult(
            product_name=product_name,
            category=result_data.get('category', 'OTHER'),
            confidence=float(result_data.get('confidence', 0.5)),
            reasoning=result_data.get('reasoning', 'Category classification completed')
        )

    def _create_fallback_result(self, item: Dict[str, Any], index: int) -> CategoryResult:
        """Create fallback result when parsing fails"""
        product_name = (
            item.get('Product Name') or
            item.get('product_name') or
            f"Furniture Item {index + 1}"
        )

        # Try to guess category from existing data
        category = item.get('Category', item.get('Type', 'OTHER')).upper()

        return CategoryResult(
            product_name=product_name,
            category=category,
            confidence=0.3,
            reasoning="Could not parse classification response, used existing data"
        )

    def _create_error_result(self, item: Dict[str, Any], index: int, error_msg: str) -> CategoryResult:
        """Create error result"""
        product_name = (
            item.get('Product Name') or
            item.get('product_name') or
            f"Furniture Item {index + 1}"
        )

        return CategoryResult(
            product_name=product_name,
            category="ERROR",
            confidence=0.0,
            reasoning=error_msg
        )

# Example usage
async def main():
    # Sample furniture data
    sample_batch = [
        {
            "Product Name": "Modern Leather Sofa",
            "Description": "High-quality leather sofa with minimalist design",
            "Material": "Genuine leather",
            "Color": "Black",
            "Dimensions": "84\"W x 36\"D x 30\"H"
        },
        {
            "Product Name": "Vintage Oak Dining Table",
            "Description": "Solid oak table with distressed finish",
            "Material": "Oak wood",
            "Color": "Natural wood",
            "Dimensions": "72\"L x 36\"W x 30\"H"
        }
    ]

    classifier = TextBasedFurnitureClassifier()
    result = await classifier.classify_text_batch(sample_batch, batch_id=1)

    print("\nClassification Results:")
    for item_result in result.results:
        print(f"\nProduct: {item_result.product_name}")
        print(f"Category: {item_result.category}")
        print(f"Confidence: {item_result.confidence}")
        print(f"Reasoning: {item_result.reasoning}")

if __name__ == "__main__":
    asyncio.run(main())

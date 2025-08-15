"""OpenAI Description Service for generating image and text descriptions."""

import logging
import openai
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DescriptionService:
    """Service for generating descriptions using OpenAI."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_image_description(self, image_url: str) -> Optional[str]:
        """
        Generate a detailed description of an image using OpenAI Vision API.
        
        Args:
            image_url: URL of the image to describe
            
        Returns:
            String description of the image, or None if failed
        """
        try:
            if not image_url:
                logger.warning("No image URL provided for description")
                return None
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Provide a concise, detailed description of this furniture item in exactly 3-4 sentences. Focus on:
                                - Physical appearance and key design elements
                                - Materials and style characteristics
                                - Color scheme and overall aesthetic appeal

                                Keep it descriptive but brief, suitable for search and categorization."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            description = response.choices[0].message.content
            if description:
                description = description.strip()
                logger.info(f"Generated image description: {description[:100]}...")
                return description
            else:
                logger.warning("Empty description returned from OpenAI")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image description: {str(e)}")
            return None
    
    def generate_image_description_from_base64(self, base64_image: str) -> Optional[str]:
        """
        Generate a detailed description of an image from base64 data.
        
        Args:
            base64_image: Base64 encoded image data
            
        Returns:
            String description of the image, or None if failed
        """
        try:
            if not base64_image:
                logger.warning("No base64 image data provided for description")
                return None
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Provide a concise, detailed description of this furniture item in exactly 3-4 sentences. Focus on:
                                - Physical appearance and key design elements
                                - Materials and style characteristics
                                - Color scheme and overall aesthetic appeal

                                Keep it descriptive but brief, suitable for search and categorization."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            description = response.choices[0].message.content
            if description:
                description = description.strip()
                logger.info(f"Generated image description from base64: {description[:100]}...")
                return description
            else:
                logger.warning("Empty description returned from OpenAI")
                return None
                
        except Exception as e:
            logger.error(f"Error generating image description from base64: {str(e)}")
            return None
    
    def generate_text_description(self, product_info: dict) -> Optional[str]:
        """
        Generate a description from product text information.
        
        Args:
            product_info: Dictionary containing product information
            
        Returns:
            String description, or None if failed
        """
        try:
            # Build text from available product information
            text_parts = []
            
            if product_info.get('product_name'):
                text_parts.append(f"Product: {product_info['product_name']}")
            
            if product_info.get('type'):
                text_parts.append(f"Type: {product_info['type']}")
            
            if product_info.get('style'):
                if isinstance(product_info['style'], list):
                    style_str = ", ".join(product_info['style'])
                else:
                    style_str = str(product_info['style'])
                text_parts.append(f"Style: {style_str}")
            
            if product_info.get('tags'):
                if isinstance(product_info['tags'], list):
                    tags_str = ", ".join(product_info['tags'])
                else:
                    tags_str = str(product_info['tags'])
                text_parts.append(f"Tags: {tags_str}")
            
            if product_info.get('price_range_usd'):
                text_parts.append(f"Price Range: {product_info['price_range_usd']}")
            
            if not text_parts:
                logger.warning("No product information available for text description")
                return None
            
            product_text = " | ".join(text_parts)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a furniture description expert. Create detailed, natural descriptions of furniture items based on the provided information."
                    },
                    {
                        "role": "user",
                        "content": f"""Based on this furniture information, create a concise description in exactly 3-4 sentences that captures the item's characteristics, style, and appeal:

{product_text}

Write a natural, descriptive summary that would be useful for search and categorization."""
                    }
                ],
                max_tokens=120,
                temperature=0.3
            )
            
            description = response.choices[0].message.content
            if description:
                description = description.strip()
                logger.info(f"Generated text-based description: {description[:100]}...")
                return description
            else:
                logger.warning("Empty description returned from OpenAI")
                return None
                
        except Exception as e:
            logger.error(f"Error generating text description: {str(e)}")
            return None

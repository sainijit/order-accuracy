"""
VLM Client Service for OpenVINO Model Server interaction.
Implements Adapter pattern for VLM inference abstraction.
"""

import base64
import json
import logging
from typing import Dict, List, Any, Optional
from io import BytesIO
import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class VLMResponse:
    """Value object for VLM inference results"""
    
    def __init__(self, raw_response: Dict[str, Any]):
        self.raw_response = raw_response
        self.detected_items: List[Dict[str, Any]] = []
        self._parse_response()
    
    def _parse_response(self):
        """Parse VLM response to extract detected items"""
        try:
            # Extract content from OpenAI-compatible response
            if "choices" in self.raw_response:
                content = self.raw_response["choices"][0]["message"]["content"]
                
                # Try to parse as JSON first (structured output)
                try:
                    parsed_content = json.loads(content)
                    if isinstance(parsed_content, dict) and "items" in parsed_content:
                        self.detected_items = parsed_content["items"]
                    elif isinstance(parsed_content, list):
                        self.detected_items = parsed_content
                    else:
                        logger.warning(f"Unexpected JSON structure in VLM response: {parsed_content}")
                except json.JSONDecodeError:
                    # Fallback: parse natural language response
                    self._parse_natural_language(content)
                    
                logger.info(f"Parsed {len(self.detected_items)} items from VLM response")
            else:
                logger.error(f"Unexpected VLM response format: {self.raw_response}")
        except Exception as e:
            logger.exception(f"Error parsing VLM response: {e}")
    
    def _parse_natural_language(self, content: str):
        """Fallback parser for natural language VLM responses"""
        # Simple pattern matching for common food item descriptions
        # Format: "- item_name (quantity: N)" or "- item_name x N"
        import re
        patterns = [
            r'-\s*([^(]+)\s*\(quantity:\s*(\d+)\)',
            r'-\s*([^x]+)\s*x\s*(\d+)',
            r'(\d+)\s*x\s*([^,\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                for match in matches:
                    if len(match) == 2:
                        name, quantity = match if pattern.startswith(r'-\s*([^(]') else (match[1], match[0])
                        self.detected_items.append({
                            "name": name.strip(),
                            "quantity": int(quantity)
                        })
                break
        
        logger.info(f"Parsed {len(self.detected_items)} items from natural language")


class VLMClient:
    """
    VLM Client implementing Adapter pattern.
    Provides abstraction over OpenVINO Model Server VLM endpoint.
    """
    
    def __init__(self, endpoint: str, model_name: str, timeout: int = 60):
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout = timeout
        self.chat_endpoint = f"{endpoint}/v3/chat/completions"
        logger.info(f"VLM Client initialized: endpoint={endpoint}, model={model_name}")
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image to base64 for VLM input"""
        try:
            # Validate image can be opened
            img = Image.open(BytesIO(image_bytes))
            logger.debug(f"Image validated: format={img.format}, size={img.size}")
            
            # Encode to base64
            encoded = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            logger.exception(f"Error encoding image: {e}")
            raise
    
    def _build_prompt(self) -> str:
        """Build structured prompt for food plate analysis"""
        return """Analyze this food plate image carefully. List all food items you can identify with their quantities.

Return the result as a JSON object with this exact structure:
{
  "items": [
    {"name": "item_name", "quantity": number},
    ...
  ]
}

Be specific with item names. For example, use "french fries" instead of just "fries", "chicken burger" instead of just "burger".
Count quantities accurately."""
    
    async def analyze_plate(self, image_bytes: bytes) -> VLMResponse:
        """
        Analyze food plate image using VLM.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            VLMResponse with detected items
            
        Raises:
            httpx.HTTPError: On network or API errors
        """
        logger.info("Starting VLM analysis")
        
        try:
            # Encode image
            encoded_image = self._encode_image(image_bytes)
            
            # Build request payload (OpenAI-compatible format)
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._build_prompt()},
                            {"type": "image_url", "image_url": {"url": encoded_image}}
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1  # Low temperature for consistent structured output
            }
            
            # Make async request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.debug(f"Sending request to {self.chat_endpoint}")
                response = await client.post(
                    self.chat_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"VLM analysis completed successfully")
                logger.debug(f"Raw VLM response: {result}")
                
                return VLMResponse(result)
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during VLM analysis: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during VLM analysis: {e}")
            raise

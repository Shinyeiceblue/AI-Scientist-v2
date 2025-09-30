"""
Model fallback utilities for improved reliability when working with multiple AI providers.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from ai_scientist import llm, vlm

logger = logging.getLogger("ai-scientist")


class ModelFallbackManager:
    """Manages fallback between different AI models for improved reliability."""
    
    def __init__(self):
        self.model_groups = {
            "reasoning": [
                "o1-preview-2024-09-12",
                "o3-mini-2025-01-31", 
                "claude-3-5-sonnet-20241022",
                "gpt-4o-2024-11-20",
            ],
            "coding": [
                "claude-3-5-sonnet-20241022",
                "deepseek-coder-v2-0724",
                "gpt-4o-2024-11-20",
                "ollama/deepseek-coder-v2",
                "ollama/codellama:34b",
            ],
            "general": [
                "gpt-4o-2024-11-20",
                "claude-3-5-sonnet-20241022",
                "gemini-2.0-flash",
                "ollama/llama3.1:8b",
            ],
            "vision": [
                "gpt-4o-2024-11-20",
                "claude-3-5-sonnet-20241022",
                "gpt-4o-mini-2024-07-18",
                "gemini-2.0-flash",
            ],
            "cheap": [
                "gpt-4o-mini-2024-07-18", 
                "claude-3-haiku-20240307",
                "gemini-2.0-flash",
                "ollama/llama3.1:8b",
            ],
        }
        
        # Track failed models to avoid repeated failures
        self.failed_models = set()
        
    def get_fallback_models(self, preferred_model: str, task_type: str = "general") -> List[str]:
        """Get a list of models to try, starting with the preferred model."""
        fallback_list = [preferred_model]
        
        # Add models from the appropriate group
        if task_type in self.model_groups:
            for model in self.model_groups[task_type]:
                if model not in fallback_list and model not in self.failed_models:
                    fallback_list.append(model)
        
        # Add general fallbacks if not already included
        for model in self.model_groups["general"]:
            if model not in fallback_list and model not in self.failed_models:
                fallback_list.append(model)
                
        return fallback_list
        
    def mark_model_failed(self, model: str, error: Exception):
        """Mark a model as temporarily failed."""
        logger.warning(f"Marking model {model} as failed due to error: {error}")
        self.failed_models.add(model)
        
    def clear_failed_models(self):
        """Clear the list of failed models (call periodically)."""
        self.failed_models.clear()
        
    def get_response_with_fallback(
        self,
        prompt: str,
        system_message: str,
        preferred_model: str,
        task_type: str = "general",
        **kwargs
    ) -> Tuple[str, str]:
        """
        Get response using model fallback mechanism.
        
        Returns:
            Tuple of (response, model_used)
        """
        fallback_models = self.get_fallback_models(preferred_model, task_type)
        last_error = None
        
        for model in fallback_models:
            try:
                client, model_name = llm.create_client(model)
                response, _ = llm.get_response_from_llm(
                    prompt=prompt,
                    client=client,
                    model=model_name,
                    system_message=system_message,
                    **kwargs
                )
                logger.info(f"Successfully used model: {model}")
                return response, model
                
            except Exception as e:
                logger.warning(f"Model {model} failed with error: {e}")
                self.mark_model_failed(model, e)
                last_error = e
                continue
                
        # If all models fail, raise the last error
        raise Exception(f"All fallback models failed. Last error: {last_error}")
        
    def get_vlm_response_with_fallback(
        self,
        msg: str,
        image_paths: Any,
        system_message: str,
        preferred_model: str,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Get VLM response using model fallback mechanism.
        
        Returns:
            Tuple of (response, model_used)
        """
        fallback_models = self.get_fallback_models(preferred_model, "vision")
        # Filter to only VLM models
        fallback_models = [m for m in fallback_models if m in vlm.AVAILABLE_VLMS]
        last_error = None
        
        for model in fallback_models:
            try:
                client, model_name = vlm.create_client(model)
                response, _ = vlm.get_response_from_vlm(
                    msg=msg,
                    image_paths=image_paths,
                    client=client,
                    model=model_name,
                    system_message=system_message,
                    **kwargs
                )
                logger.info(f"Successfully used VLM model: {model}")
                return response, model
                
            except Exception as e:
                logger.warning(f"VLM model {model} failed with error: {e}")
                self.mark_model_failed(model, e)
                last_error = e
                continue
                
        # If all models fail, raise the last error
        raise Exception(f"All VLM fallback models failed. Last error: {last_error}")


# Global instance
fallback_manager = ModelFallbackManager()


def get_response_with_fallback(
    prompt: str,
    system_message: str,
    preferred_model: str,
    task_type: str = "general",
    **kwargs
) -> Tuple[str, str]:
    """Convenience function to get response with fallback."""
    return fallback_manager.get_response_with_fallback(
        prompt, system_message, preferred_model, task_type, **kwargs
    )


def get_vlm_response_with_fallback(
    msg: str,
    image_paths: Any,
    system_message: str,
    preferred_model: str,
    **kwargs
) -> Tuple[str, str]:
    """Convenience function to get VLM response with fallback."""
    return fallback_manager.get_vlm_response_with_fallback(
        msg, image_paths, system_message, preferred_model, **kwargs
    )
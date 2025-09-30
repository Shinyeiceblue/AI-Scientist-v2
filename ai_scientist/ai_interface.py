"""
Unified AI Provider Interface for working with different AI models seamlessly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import logging
from ai_scientist import llm, vlm
from ai_scientist.model_fallback import get_response_with_fallback, get_vlm_response_with_fallback

logger = logging.getLogger("ai-scientist")


@dataclass
class AIRequest:
    """Represents a request to an AI model."""
    prompt: str
    system_message: str = ""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    model: Optional[str] = None
    task_type: str = "general"
    
    
@dataclass
class VisionRequest:
    """Represents a request to a vision-language model."""
    prompt: str
    image_paths: Union[str, List[str]]
    system_message: str = ""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    model: Optional[str] = None


@dataclass
class AIResponse:
    """Represents a response from an AI model."""
    content: str
    model_used: str
    success: bool = True
    error: Optional[str] = None


class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    @abstractmethod
    def generate_text(self, request: AIRequest) -> AIResponse:
        """Generate text response from AI model."""
        pass
        
    @abstractmethod
    def generate_vision_response(self, request: VisionRequest) -> AIResponse:
        """Generate response from vision-language model."""
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available/configured."""
        pass


class UnifiedAIProvider(AIProvider):
    """Unified AI provider that works with all supported models."""
    
    def __init__(self, 
                 default_text_model: str = "gpt-4o-2024-11-20",
                 default_vision_model: str = "gpt-4o-2024-11-20",
                 use_fallback: bool = True):
        self.default_text_model = default_text_model
        self.default_vision_model = default_vision_model
        self.use_fallback = use_fallback
        
    def generate_text(self, request: AIRequest) -> AIResponse:
        """Generate text response using fallback mechanism if enabled."""
        model = request.model or self.default_text_model
        
        try:
            if self.use_fallback:
                response, model_used = get_response_with_fallback(
                    prompt=request.prompt,
                    system_message=request.system_message,
                    preferred_model=model,
                    task_type=request.task_type,
                    temperature=request.temperature
                )
            else:
                client, model_name = llm.create_client(model)
                response, _ = llm.get_response_from_llm(
                    prompt=request.prompt,
                    client=client,
                    model=model_name,
                    system_message=request.system_message,
                    temperature=request.temperature
                )
                model_used = model
                
            return AIResponse(
                content=response,
                model_used=model_used,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to generate text response: {e}")
            return AIResponse(
                content="",
                model_used=model,
                success=False,
                error=str(e)
            )
    
    def generate_vision_response(self, request: VisionRequest) -> AIResponse:
        """Generate vision response using fallback mechanism if enabled."""
        model = request.model or self.default_vision_model
        
        try:
            if self.use_fallback:
                response, model_used = get_vlm_response_with_fallback(
                    msg=request.prompt,
                    image_paths=request.image_paths,
                    system_message=request.system_message,
                    preferred_model=model,
                    temperature=request.temperature
                )
            else:
                client, model_name = vlm.create_client(model)
                response, _ = vlm.get_response_from_vlm(
                    msg=request.prompt,
                    image_paths=request.image_paths,
                    client=client,
                    model=model_name,
                    system_message=request.system_message,
                    temperature=request.temperature
                )
                model_used = model
                
            return AIResponse(
                content=response,
                model_used=model_used,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to generate vision response: {e}")
            return AIResponse(
                content="",
                model_used=model,
                success=False,
                error=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if the provider is available."""
        try:
            # Try to create a client with the default model
            llm.create_client(self.default_text_model)
            return True
        except Exception:
            return False


class TaskSpecificAI:
    """High-level interface for task-specific AI operations."""
    
    def __init__(self, provider: AIProvider = None):
        self.provider = provider or UnifiedAIProvider()
        
    def generate_code(self, problem: str, language: str = "python", model: str = None) -> AIResponse:
        """Generate code for a given problem."""
        system_msg = f"""You are an expert {language} programmer. 
Generate clean, well-commented, and efficient code to solve the given problem.
Follow best practices and include error handling where appropriate."""
        
        request = AIRequest(
            prompt=problem,
            system_message=system_msg,
            model=model,
            task_type="coding",
            temperature=0.1  # Lower temperature for more deterministic code
        )
        return self.provider.generate_text(request)
        
    def analyze_research(self, content: str, model: str = None) -> AIResponse:
        """Analyze research content."""
        system_msg = """You are a research analyst. Analyze the given content and provide insights,
identify key findings, methodology strengths and weaknesses, and suggest improvements."""
        
        request = AIRequest(
            prompt=content,
            system_message=system_msg,
            model=model,
            task_type="reasoning",
            temperature=0.3
        )
        return self.provider.generate_text(request)
        
    def summarize_text(self, text: str, model: str = None) -> AIResponse:
        """Summarize the given text."""
        system_msg = """You are an expert at summarizing text. Create a concise, 
comprehensive summary that captures the key points and main ideas."""
        
        request = AIRequest(
            prompt=f"Please summarize the following text:\n\n{text}",
            system_message=system_msg,
            model=model,
            task_type="general",
            temperature=0.3
        )
        return self.provider.generate_text(request)
        
    def analyze_images(self, image_paths: Union[str, List[str]], question: str, model: str = None) -> AIResponse:
        """Analyze images and answer questions about them."""
        system_msg = """You are an expert at analyzing images. Provide detailed, accurate 
descriptions and answer questions about the visual content."""
        
        request = VisionRequest(
            prompt=question,
            image_paths=image_paths,
            system_message=system_msg,
            model=model,
            temperature=0.3
        )
        return self.provider.generate_vision_response(request)
        
    def brainstorm_ideas(self, topic: str, num_ideas: int = 5, model: str = None) -> AIResponse:
        """Brainstorm creative ideas on a given topic."""
        system_msg = """You are a creative ideation expert. Generate innovative, 
practical, and diverse ideas that could lead to breakthrough solutions."""
        
        prompt = f"""Generate {num_ideas} creative and innovative ideas related to: {topic}

For each idea, provide:
1. A clear title
2. A brief description
3. Potential benefits
4. Implementation considerations"""
        
        request = AIRequest(
            prompt=prompt,
            system_message=system_msg,
            model=model,
            task_type="general",
            temperature=0.8  # Higher temperature for creativity
        )
        return self.provider.generate_text(request)


# Global instances for easy access
unified_ai = UnifiedAIProvider()
task_ai = TaskSpecificAI(unified_ai)


# Convenience functions
def generate_code(problem: str, language: str = "python", model: str = None) -> AIResponse:
    """Generate code for a given problem."""
    return task_ai.generate_code(problem, language, model)


def analyze_research(content: str, model: str = None) -> AIResponse:
    """Analyze research content."""  
    return task_ai.analyze_research(content, model)


def summarize_text(text: str, model: str = None) -> AIResponse:
    """Summarize the given text."""
    return task_ai.summarize_text(text, model)


def analyze_images(image_paths: Union[str, List[str]], question: str, model: str = None) -> AIResponse:
    """Analyze images and answer questions about them."""
    return task_ai.analyze_images(image_paths, question, model)


def brainstorm_ideas(topic: str, num_ideas: int = 5, model: str = None) -> AIResponse:
    """Brainstorm creative ideas on a given topic."""
    return task_ai.brainstorm_ideas(topic, num_ideas, model)
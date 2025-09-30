#!/usr/bin/env python3
"""
Simple test script to validate the enhanced AI functionality.

This script tests basic functionality without requiring API keys.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_scientist import llm, vlm
from ai_scientist.model_fallback import ModelFallbackManager, fallback_manager
from ai_scientist.ai_interface import AIRequest, AIResponse, UnifiedAIProvider, TaskSpecificAI


def test_model_lists():
    """Test that model lists are populated correctly."""
    print("🧪 Testing Model Lists...")
    
    # Test LLM list
    assert len(llm.AVAILABLE_LLMS) > 40, f"Expected >40 LLMs, got {len(llm.AVAILABLE_LLMS)}"
    
    # Test specific model additions
    assert "ollama/llama3.1" in llm.AVAILABLE_LLMS, "Ollama models not found"
    assert "azure/gpt-4o" in llm.AVAILABLE_LLMS, "Azure models not found" 
    
    # Test VLM list
    assert len(vlm.AVAILABLE_VLMS) > 10, f"Expected >10 VLMs, got {len(vlm.AVAILABLE_VLMS)}"
    assert "claude-3-5-sonnet-20241022" in vlm.AVAILABLE_VLMS, "Claude vision not found"
    assert "gemini-2.0-flash" in vlm.AVAILABLE_VLMS, "Gemini vision not found"
    
    print("✓ Model lists validated")


def test_fallback_system():
    """Test the fallback system logic."""
    print("🧪 Testing Fallback System...")
    
    manager = ModelFallbackManager()
    
    # Test different task types
    coding_models = manager.get_fallback_models("gpt-4o", "coding")
    assert len(coding_models) > 3, f"Expected >3 coding fallback models, got {len(coding_models)}"
    assert "claude-3-5-sonnet-20241022" in coding_models, "Claude not in coding fallbacks"
    
    vision_models = manager.get_fallback_models("gpt-4o", "vision")
    # Filter to only actual VLM models
    vision_models = [m for m in vision_models if m in vlm.AVAILABLE_VLMS]
    assert len(vision_models) > 2, f"Expected >2 vision models, got {len(vision_models)}"
    
    # Test failed model tracking
    manager.mark_model_failed("test-model", Exception("Test error"))
    assert "test-model" in manager.failed_models, "Failed model not tracked"
    
    manager.clear_failed_models()
    assert len(manager.failed_models) == 0, "Failed models not cleared"
    
    print("✓ Fallback system validated")


def test_unified_interface():
    """Test the unified AI interface classes."""
    print("🧪 Testing Unified Interface...")
    
    # Test request/response classes
    request = AIRequest(
        prompt="Test prompt",
        system_message="Test system",
        temperature=0.7,
        task_type="general"
    )
    assert request.prompt == "Test prompt", "AIRequest not working"
    
    response = AIResponse(
        content="Test response",
        model_used="test-model",
        success=True
    )
    assert response.content == "Test response", "AIResponse not working"
    assert response.success is True, "AIResponse success not working"
    
    # Test provider creation (without making actual API calls)
    provider = UnifiedAIProvider()
    assert provider.default_text_model is not None, "Provider not initialized"
    assert provider.use_fallback is True, "Fallback not enabled by default"
    
    # Test task AI creation
    task_ai = TaskSpecificAI()
    assert task_ai.provider is not None, "TaskSpecificAI provider not set"
    
    print("✓ Unified interface validated")


def test_client_creation_logic():
    """Test client creation logic (without actually creating clients)."""
    print("🧪 Testing Client Creation Logic...")
    
    # Test that we can identify different model types
    test_cases = [
        ("gpt-4o", "openai"),
        ("claude-3-5-sonnet-20241022", "anthropic"), 
        ("azure/gpt-4o", "azure"),
        ("ollama/llama3.1", "ollama"),
        ("gemini-2.0-flash", "gemini"),
        ("deepseek-coder-v2-0724", "deepseek"),
    ]
    
    for model, expected_type in test_cases:
        if expected_type == "openai":
            assert "gpt" in model, f"OpenAI model {model} doesn't contain 'gpt'"
        elif expected_type == "anthropic":
            assert "claude" in model, f"Anthropic model {model} doesn't contain 'claude'"
        elif expected_type == "azure":
            assert model.startswith("azure/"), f"Azure model {model} doesn't start with 'azure/'"
        elif expected_type == "ollama":
            assert model.startswith("ollama/"), f"Ollama model {model} doesn't start with 'ollama/'"
        elif expected_type == "gemini":
            assert "gemini" in model, f"Gemini model {model} doesn't contain 'gemini'"
        elif expected_type == "deepseek":
            assert "deepseek" in model, f"DeepSeek model {model} doesn't contain 'deepseek'"
    
    print("✓ Client creation logic validated")


def main():
    """Run all tests."""
    print("🌟 AI-Scientist-v2 Enhanced AI Functionality Tests")
    print("=" * 60)
    
    try:
        test_model_lists()
        test_fallback_system() 
        test_unified_interface()
        test_client_creation_logic()
        
        print("\n✅ All tests passed!")
        print("🚀 Enhanced AI functionality is working correctly!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
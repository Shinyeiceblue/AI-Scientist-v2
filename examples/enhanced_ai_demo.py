#!/usr/bin/env python3
"""
Example script demonstrating the enhanced AI capabilities in AI-Scientist-v2.

This script showcases:
1. Support for multiple AI providers (OpenAI, Anthropic, Ollama, Azure, etc.)
2. Vision-language model capabilities with multiple providers
3. Automatic model fallback mechanisms
4. Unified AI interface for various tasks
5. Task-specific AI helpers

Usage:
    python examples/enhanced_ai_demo.py
"""

import sys
import os

# Add the parent directory to path so we can import ai_scientist modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_scientist import llm, vlm
from ai_scientist.model_fallback import ModelFallbackManager
from ai_scientist.ai_interface import TaskSpecificAI, AIRequest, VisionRequest, UnifiedAIProvider


def demo_model_availability():
    """Demonstrate the wide range of supported AI models."""
    print("🤖 Enhanced AI Model Support in AI-Scientist-v2")
    print("=" * 60)
    
    print(f"📊 Total Language Models: {len(llm.AVAILABLE_LLMS)}")
    print(f"👀 Total Vision Models: {len(vlm.AVAILABLE_VLMS)}")
    print()
    
    print("🔧 Supported Providers:")
    providers = {
        "OpenAI": [m for m in llm.AVAILABLE_LLMS if "gpt" in m or "o1" in m or "o3" in m],
        "Anthropic": [m for m in llm.AVAILABLE_LLMS if "claude" in m],
        "Google": [m for m in llm.AVAILABLE_LLMS if "gemini" in m],
        "Ollama (Local)": [m for m in llm.AVAILABLE_LLMS if m.startswith("ollama/")],
        "Azure OpenAI": [m for m in llm.AVAILABLE_LLMS if m.startswith("azure/")],
        "DeepSeek": [m for m in llm.AVAILABLE_LLMS if "deepseek" in m],
        "HuggingFace": [m for m in llm.AVAILABLE_LLMS if "deepcoder" in m],
    }
    
    for provider, models in providers.items():
        print(f"  • {provider}: {len(models)} models")
        if models:
            print(f"    Examples: {', '.join(models[:2])}")
    print()


def demo_fallback_system():
    """Demonstrate the model fallback system."""
    print("🔄 Model Fallback System")
    print("=" * 60)
    
    fallback_manager = ModelFallbackManager()
    
    task_types = ["coding", "reasoning", "general", "vision", "cheap"]
    for task_type in task_types:
        fallback_models = fallback_manager.get_fallback_models("gpt-4o-2024-11-20", task_type)
        print(f"📋 {task_type.title()} Tasks: {len(fallback_models)} fallback models")
        print(f"   Priority: {' → '.join(fallback_models[:3])}...")
    print()


def demo_unified_interface():
    """Demonstrate the unified AI interface."""
    print("🎯 Unified AI Interface")
    print("=" * 60)
    
    # Create task-specific AI instance
    task_ai = TaskSpecificAI()
    
    print("🚀 Available Task-Specific Functions:")
    functions = [
        "generate_code(problem, language)",
        "analyze_research(content)",
        "summarize_text(text)", 
        "analyze_images(image_paths, question)",
        "brainstorm_ideas(topic, num_ideas)"
    ]
    
    for func in functions:
        print(f"  • {func}")
    
    print("\n📝 Example Usage:")
    print("""
# Generate Python code
response = task_ai.generate_code("Create a fibonacci function", "python")

# Analyze research with fallback
response = task_ai.analyze_research(paper_content, model="claude-3-5-sonnet-20241022")

# Analyze images with vision models
response = task_ai.analyze_images(["chart.png", "graph.jpg"], "What trends do you see?")

# Get creative ideas with high temperature
response = task_ai.brainstorm_ideas("AI-assisted scientific discovery", num_ideas=10)
""")


def demo_provider_examples():
    """Show examples of different provider configurations."""
    print("⚙️ Provider Configuration Examples")
    print("=" * 60)
    
    print("🔑 Environment Variables Required:")
    env_vars = {
        "OpenAI": ["OPENAI_API_KEY"],
        "Anthropic": ["ANTHROPIC_API_KEY"],
        "Azure OpenAI": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        "Google Gemini": ["GEMINI_API_KEY"],
        "DeepSeek": ["DEEPSEEK_API_KEY"],
        "HuggingFace": ["HUGGINGFACE_API_KEY"],
        "Ollama": ["OLLAMA_BASE_URL (optional, defaults to localhost:11434)"],
    }
    
    for provider, vars in env_vars.items():
        print(f"  • {provider}: {', '.join(vars)}")
    
    print("\n🐳 Ollama Setup (Local AI):")
    print("""
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.1:8b
ollama pull codellama:7b
ollama pull deepseek-coder-v2

# Start server (runs on http://localhost:11434 by default)
ollama serve
""")


def demo_vision_capabilities():
    """Demonstrate enhanced vision capabilities."""
    print("👁️ Enhanced Vision-Language Models")
    print("=" * 60)
    
    print("🎨 Supported Vision Models:")
    vision_groups = {
        "OpenAI": [m for m in vlm.AVAILABLE_VLMS if "gpt" in m or "o3" in m],
        "Anthropic": [m for m in vlm.AVAILABLE_VLMS if "claude" in m],
        "Google": [m for m in vlm.AVAILABLE_VLMS if "gemini" in m],
    }
    
    for provider, models in vision_groups.items():
        if models:
            print(f"  • {provider}: {', '.join(models)}")
    
    print("\n🖼️ Example Vision Tasks:")
    print("""
# Analyze scientific charts
response = vlm.get_response_from_vlm(
    "Describe the trends in this research data",
    image_paths=["results_chart.png"],
    model="claude-3-5-sonnet-20241022",
    system_message="You are a scientific data analyst"
)

# Multi-image analysis
response = vlm.get_response_from_vlm(
    "Compare these experimental setups",
    image_paths=["setup1.jpg", "setup2.jpg", "setup3.jpg"],
    model="gpt-4o-2024-11-20"
)
""")


def main():
    """Run the demonstration."""
    print("🌟 AI-Scientist-v2: Enhanced AI Model Support Demo")
    print("=" * 80)
    print()
    
    demo_model_availability()
    print()
    
    demo_fallback_system()
    print()
    
    demo_unified_interface()
    print()
    
    demo_provider_examples()
    print()
    
    demo_vision_capabilities()
    print()
    
    print("✨ Summary of Enhancements:")
    print("  • 49 language models across 7+ providers")
    print("  • 13 vision-language models")
    print("  • Automatic model fallback system")
    print("  • Unified interface for common tasks")
    print("  • Local AI support via Ollama")
    print("  • Enterprise Azure OpenAI support")
    print("  • Enhanced vision capabilities")
    print()
    print("🚀 The AI-Scientist-v2 is now ready to utilize various AI models!")


if __name__ == "__main__":
    main()
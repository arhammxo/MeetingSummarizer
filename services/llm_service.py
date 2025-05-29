"""
LLM Service - Factory for creating LLM instances based on configuration
"""
import logging
from typing import Optional, Dict, Any
from config import settings

logger = logging.getLogger("llm-service")

def get_llm(temperature: float = 0, model_name: Optional[str] = None, purpose: str = "general"):
    """
    Factory function to get a configured LLM based on settings
    
    Args:
        temperature: Temperature setting for the LLM
        model_name: Specific model to use (overrides defaults)
        purpose: What the LLM will be used for ("general", "summarization", "multilingual")
        
    Returns:
        A configured LLM instance
    """
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "openai":
        return get_openai_llm(temperature, model_name)
    elif provider == "ollama":
        return get_ollama_llm(temperature, model_name, purpose)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def get_openai_llm(temperature: float = 0, model_name: Optional[str] = None):
    """Get an OpenAI LLM instance"""
    try:
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=model_name or "gpt-4o",
            temperature=temperature
        )
    except ImportError:
        logger.error("langchain_openai not installed. Run: pip install langchain-openai")
        raise

def get_ollama_llm(temperature: float = 0, model_name: Optional[str] = None, purpose: str = "general", format_schema: Optional[Dict] = None):
    """Get an Ollama LLM instance with structured output support"""
    try:
        from langchain_ollama import ChatOllama
        
        # Select the appropriate model based on purpose
        if not model_name:
            if purpose == "summarization":
                model_name = settings.OLLAMA_SUMMARIZATION_MODEL
            elif purpose == "multilingual":
                model_name = settings.OLLAMA_MULTILINGUAL_MODEL
            else:
                model_name = settings.OLLAMA_MODEL
        
        logger.info(f"Creating Ollama LLM with model: {model_name}, purpose: {purpose}")
        
        # Use ChatOllama for better structured output support
        llm = ChatOllama(
            base_url=settings.OLLAMA_API_BASE,
            model=model_name,
            temperature=temperature,
            format=format_schema,  # Add structured output support
            timeout=120.0,
            num_predict=4096,  # Increase token limit
        )
        
        return llm
        
    except ImportError:
        logger.error("langchain_ollama not installed. Run: pip install langchain-ollama")
        raise

def create_chat_prompt_template(system_prompt, user_prompt, use_simple_format=False):
    """Create a chat prompt template with Ollama optimization"""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        
        if settings.LLM_PROVIDER == "ollama" and use_simple_format:
            # Simplified format for Ollama
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            return ChatPromptTemplate.from_template(combined_prompt)
        else:
            # Standard format
            return ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
    except ImportError:
        logger.error("langchain_core not installed")
        raise

def create_output_parser():
    """
    Create an appropriate output parser based on the LLM provider
    
    Returns:
        A configured output parser
    """
    try:
        from langchain_core.output_parsers import JsonOutputParser
        
        return JsonOutputParser()
    except ImportError:
        logger.error("langchain_core not installed. Run: pip install langchain-core")
        raise
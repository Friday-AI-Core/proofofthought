"""Azure OpenAI Configuration Helper

This module provides easy configuration for Azure OpenAI GPT-5,
with support for environment variables and multiple deployment options.

Environment Variables:
    AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL (required)
    AZURE_OPENAI_KEY: Azure OpenAI API key (required)
    AZURE_DEPLOYMENT_NAME: Deployment name (optional, default: 'gpt-5')
    AZURE_API_VERSION: API version (optional, default: '2024-12-01-preview')
    AZURE_MAX_TOKENS: Max tokens for completions (optional, default: 16384)
    AZURE_TEMPERATURE: Temperature for completions (optional, default: 0.1)

Multiple Deployments:
    To use multiple deployments/models, you can:
    1. Set different AZURE_DEPLOYMENT_NAME values per environment
    2. Use get_azure_client() with custom deployment names
    3. Call get_client_config() with custom model parameter

Usage:
    1. Copy .env.example to .env in the project root
    2. Fill in your Azure OpenAI credentials in .env
    3. Import and use:
       - get_azure_client() to get configured client
       - get_client_config() to get full configuration
       - is_configured() to check if minimum config is set
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from openai import AzureOpenAI

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
_ENV_LOADED = False
_ENV_LOAD_ERROR = None

try:
    from dotenv import load_dotenv
    
    # Load from project root .env file
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        _ENV_LOADED = True
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}. Using system environment variables.")
        _ENV_LOADED = False
except ImportError as e:
    _ENV_LOAD_ERROR = str(e)
    logger.warning(
        "python-dotenv not installed. Install with: pip install python-dotenv\n"
        "Falling back to system environment variables only."
    )

# Azure OpenAI Configuration
# Loaded from environment variables with sensible defaults where applicable
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-5")
API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

# GPT-5 specific settings with environment variable override support
_DEFAULT_MAX_TOKENS = 16384
_DEFAULT_TEMPERATURE = 0.1

GPT5_MAX_TOKENS = int(os.getenv("AZURE_MAX_TOKENS", str(_DEFAULT_MAX_TOKENS)))
GPT5_TEMPERATURE = float(os.getenv("AZURE_TEMPERATURE", str(_DEFAULT_TEMPERATURE)))


def is_configured() -> bool:
    """Check if minimum Azure OpenAI configuration is set.
    
    Returns:
        True if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY are set, False otherwise
    
    Example:
        >>> if is_configured():
        ...     client = get_azure_client()
        ... else:
        ...     print("Azure OpenAI is not configured")
    """
    return bool(AZURE_ENDPOINT and AZURE_API_KEY)


def get_azure_client() -> AzureOpenAI:
    """Get configured Azure OpenAI client for GPT-5.
    
    Returns:
        Configured AzureOpenAI client
    
    Environment Variables (required):
        AZURE_OPENAI_ENDPOINT: Azure endpoint URL
        AZURE_OPENAI_KEY: API key
        
    Environment Variables (optional):
        AZURE_DEPLOYMENT_NAME: Deployment name (default: 'gpt-5')
        AZURE_API_VERSION: API version (default: '2024-12-01-preview')
    
    Raises:
        ValueError: If required environment variables are not set
    
    Example:
        >>> client = get_azure_client()
        >>> response = client.chat.completions.create(
        ...     model="gpt-5",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """
    if not AZURE_ENDPOINT:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT is not set.\n"
            "\n"
            "To fix this:\n"
            "1. Create a .env file in your project root (copy from .env.example if available)\n"
            "2. Add: AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/\n"
            "3. Alternatively, set the environment variable in your system\n"
            "\n"
            f"Current .env loading status: {'Loaded' if _ENV_LOADED else 'Not loaded'}\n"
            f".env load error: {_ENV_LOAD_ERROR or 'None'}"
        )
    
    if not AZURE_API_KEY:
        raise ValueError(
            "AZURE_OPENAI_KEY is not set.\n"
            "\n"
            "To fix this:\n"
            "1. Create a .env file in your project root (copy from .env.example if available)\n"
            "2. Add: AZURE_OPENAI_KEY=your-api-key-here\n"
            "3. Alternatively, set the environment variable in your system\n"
            "4. Find your API key in the Azure Portal under your OpenAI resource\n"
            "\n"
            f"Current .env loading status: {'Loaded' if _ENV_LOADED else 'Not loaded'}\n"
            f".env load error: {_ENV_LOAD_ERROR or 'None'}"
        )
    
    logger.info(f"Creating Azure OpenAI client with endpoint: {AZURE_ENDPOINT}")
    logger.debug(f"API Version: {API_VERSION}, Deployment: {DEPLOYMENT_NAME}")
    
    return AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )


def get_client_config(
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict[str, Any]:
    """Get client configuration dictionary.
    
    Args:
        model: Override deployment name (default: uses DEPLOYMENT_NAME from env)
        max_tokens: Override max tokens (default: uses GPT5_MAX_TOKENS from env or 16384)
        temperature: Override temperature (default: uses GPT5_TEMPERATURE from env or 0.1)
    
    Returns:
        Dictionary with client, model, and recommended settings:
        {
            'llm_client': AzureOpenAI client instance,
            'model': deployment name,
            'max_tokens': maximum tokens for completion,
            'temperature': sampling temperature
        }
    
    Example:
        >>> # Use default configuration
        >>> config = get_client_config()
        >>> 
        >>> # Override specific settings
        >>> config = get_client_config(
        ...     model="gpt-4",
        ...     temperature=0.7,
        ...     max_tokens=2048
        ... )
    """
    return {
        "llm_client": get_azure_client(),
        "model": model or DEPLOYMENT_NAME,
        "max_tokens": max_tokens if max_tokens is not None else GPT5_MAX_TOKENS,
        "temperature": temperature if temperature is not None else GPT5_TEMPERATURE,
    }


# Diagnostic functions for troubleshooting
def print_diagnostics() -> None:
    """Print diagnostic information about current configuration.
    
    Useful for troubleshooting configuration issues.
    """
    print("=" * 60)
    print("Azure OpenAI Configuration Diagnostics")
    print("=" * 60)
    print()
    print("Environment Loading:")
    print(f"  .env file loaded: {_ENV_LOADED}")
    if _ENV_LOAD_ERROR:
        print(f"  Load error: {_ENV_LOAD_ERROR}")
    print()
    print("Configuration Status:")
    print(f"  Is configured: {is_configured()}")
    print()
    print("Settings:")
    print(f"  Endpoint: {AZURE_ENDPOINT or '❌ NOT SET'}")
    print(f"  API Key: {'✓ SET' if AZURE_API_KEY else '❌ NOT SET'}")
    print(f"  Deployment: {DEPLOYMENT_NAME}")
    print(f"  API Version: {API_VERSION}")
    print(f"  Max Tokens: {GPT5_MAX_TOKENS}")
    print(f"  Temperature: {GPT5_TEMPERATURE}")
    print()
    print("Environment Variables Checked:")
    print("  - AZURE_OPENAI_ENDPOINT")
    print("  - AZURE_OPENAI_KEY")
    print("  - AZURE_DEPLOYMENT_NAME")
    print("  - AZURE_API_VERSION")
    print("  - AZURE_MAX_TOKENS")
    print("  - AZURE_TEMPERATURE")
    print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for example
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )
    
    print("Azure OpenAI GPT-5 Configuration Test")
    print()
    
    # Run diagnostics
    print_diagnostics()
    print()
    
    # Test configuration check
    if is_configured():
        print("✓ Configuration check passed")
        print()
        
        try:
            # Test client creation
            print("Testing client creation...")
            client = get_azure_client()
            print("✓ Azure OpenAI client created successfully")
            print()
            
            # Show default config
            print("Default configuration:")
            config = get_client_config()
            for key, value in config.items():
                if key != "llm_client":
                    print(f"  {key}: {value}")
            print()
            
            # Show custom config example
            print("Example with custom settings:")
            custom_config = get_client_config(
                model="gpt-4",
                temperature=0.7,
                max_tokens=2048
            )
            for key, value in custom_config.items():
                if key != "llm_client":
                    print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ Configuration check failed")
        print("   Required environment variables are not set.")
        print("   Please check the diagnostics above and set the required variables.")

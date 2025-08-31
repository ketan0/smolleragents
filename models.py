from dataclasses import dataclass
from typing import Callable, List, Optional

from .core import ChatMessage, Model

def create_mock_model(
    responses: Optional[List[str]] = None,
    name: str = "mock-model",
    temperature: float = 0.7
) -> Model:
    """Create a mock model for testing."""

    if responses is None:
        responses = ["Mock response"]

    response_iter = iter(responses * 1000)  # Repeat responses

    def mock_func(messages: List[ChatMessage], **kwargs) -> ChatMessage:
        response_content = next(response_iter)
        return ChatMessage(role="assistant", content=response_content)

    return Model(
        name=name,
        func=mock_func,
        description=f"Mock model with predefined responses: {responses[:3]}{'...' if len(responses) > 3 else ''}",
        temperature=temperature
    )


def create_openai_model(
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Model:
    """Create an OpenAI model wrapper."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required for OpenAI models. Install with: pip install openai")

    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def map_role_for_openai(role: str) -> str:
        """Map internal roles to OpenAI-compatible roles"""
        role_mapping = {
            "tool-call": "assistant",
            "tool-response": "user", 
            "system": "system",
            "user": "user", 
            "assistant": "assistant"
        }
        return role_mapping.get(role, role)
    
    def openai_func(messages: List[ChatMessage], **kwargs) -> ChatMessage:
        # Convert messages to OpenAI format with role mapping
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": map_role_for_openai(msg.role),
                "content": msg.content
            })

        response = client.chat.completions.create(
            model=model_name,
            messages=openai_messages,
            **kwargs
        )

        return ChatMessage(
            role="assistant",
            content=response.choices[0].message.content
        )

    return Model(
        name=model_name,
        func=openai_func,
        description=f"OpenAI {model_name} model",
        temperature=temperature,
        max_tokens=max_tokens
    )


def create_anthropic_model(
    model_name: str = "claude-3-sonnet-20240229",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = 4096,
) -> Model:
    """Create an Anthropic model wrapper."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required for Anthropic models. Install with: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def anthropic_func(messages: List[ChatMessage], **kwargs) -> ChatMessage:
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        response = client.messages.create(
            model=model_name,
            messages=anthropic_messages,
            **kwargs
        )

        return ChatMessage(
            role="assistant",
            content=response.content[0].text
        )

    # Determine context length based on model name
    return Model(
        name=model_name,
        func=anthropic_func,
        description=f"Anthropic {model_name} model",
        temperature=temperature,
        max_tokens=max_tokens
    )


def create_basic_mock_model() -> Model:
    """Create a basic mock model for testing with reasonable defaults."""
    return create_mock_model(
        responses=[
            "I'll help you solve this step by step.",
            "Let me analyze the problem and provide a solution.",
            "Based on the information provided, here's my response."
        ],
        name="basic-mock",
    )

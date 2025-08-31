"""
Core data structures and types for functional smolagents
"""

from dataclasses import dataclass, replace, field
import json
import textwrap
from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import uuid
from dataclasses_json import dataclass_json

@dataclass(frozen=True)
class AgentConfig:
    """Immutable agent configuration"""
    max_steps: int = 20
    stream_outputs: bool = False
    use_structured_outputs: bool = False
    max_tool_threads: int = 4
    authorized_imports: List[str] = field(default_factory=lambda: ["math", "json", "re", "datetime", "random", "os"])
    code_block_tags: tuple[str, str] = ("```python", "```")
    executor_type: str = "local"
    prompt_templates: Dict[str, Any] = field(default_factory=dict)



@dataclass_json
@dataclass(frozen=True)
class ToolCall:
    """Single tool call"""
    name: str
    arguments: Union[Dict[str, Any], str]
    id: str


@dataclass(frozen=True)
class TokenUsage:
    """Token usage tracking"""
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

@dataclass(frozen=True)
class ChatMessage:
    """Immutable chat message"""
    role: str  # "system", "user", "assistant", "tool-call", "tool-response"
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    token_usage: Optional[TokenUsage] = None
    raw: Any = None



@dataclass(frozen=True)
class Timing:
    """Timing information"""
    start_time: float
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


class MemoryStep(ABC):
    """Base class for memory steps"""

    @abstractmethod
    def to_messages(self) -> List[ChatMessage]:
        pass


@dataclass(frozen=True)
class TaskStep(MemoryStep):
    """Initial task step"""
    task: str
    task_images: Optional[List[Any]] = None

    def to_messages(self) -> List[ChatMessage]:
        return [ChatMessage(role="user", content=f"New task:\n{self.task}")]


@dataclass(frozen=True)
class SystemPromptStep(MemoryStep):
    """System prompt step"""
    system_prompt: str

    def to_messages(self) -> List[ChatMessage]:
        return [ChatMessage(role="system", content=self.system_prompt)]


@dataclass(frozen=True)
class ReasoningStep(MemoryStep):
    """Model reasoning/response step"""
    model_output: str
    token_usage: Optional[TokenUsage] = None

    def to_messages(self) -> List[ChatMessage]:
        return [ChatMessage(role="assistant", content=self.model_output)]


@dataclass(frozen=True)
class ToolCallingStep(MemoryStep):
    """Tool calling step"""
    tool_calls: List[ToolCall]
    code_action: str

    def to_messages(self) -> List[ChatMessage]:
        tool_calls_str = str([tc.to_dict() for tc in self.tool_calls])
        return [ChatMessage(role="tool-call", content=f"Calling tools:\n{tool_calls_str}")]


@dataclass(frozen=True)
class ToolResponseStep(MemoryStep):
    """Tool response step"""
    observations: str
    execution_output: Any
    new_variables: Dict[str, Any]
    error: Optional[Exception] = None

    def to_messages(self) -> List[ChatMessage]:
        content = f"Observation:\n{self.observations}"
        if self.error:
            content += f"\nError: {self.error}\nRetry with care!"
        return [ChatMessage(role="tool-response", content=content)]


@dataclass(frozen=True)
class PlanningStep(MemoryStep):
    """Planning step"""
    model_input_messages: List[ChatMessage]
    model_output_message: ChatMessage
    plan: str
    token_usage: Optional[TokenUsage] = None

    def to_messages(self) -> List[ChatMessage]:
        return [
            ChatMessage(role="assistant", content=self.plan.strip()),
            ChatMessage(role="user", content="Now proceed and carry out this plan.")
        ]

@dataclass(frozen=True)
class AgentState:
    """Immutable agent state"""
    memory: List[MemoryStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    step_number: int = 0
    interrupted: bool = False

    def add_memory(self, step: MemoryStep) -> 'AgentState':
        """Add step to memory - returns new state"""
        return replace(self, memory=self.memory + [step])

    def update_variables(self, new_vars: Dict[str, Any]) -> 'AgentState':
        """Update variables - returns new state"""
        return replace(self, variables={**self.variables, **new_vars})

    def increment_step(self) -> 'AgentState':
        """Increment step number - returns new state"""
        return replace(self, step_number=self.step_number + 1)


@dataclass(frozen=True)
class Model:
    """A structured model wrapper with metadata and configuration."""

    name: str
    func: Callable
    description: str = ""
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    def __call__(self, messages: List[ChatMessage], **kwargs) -> ChatMessage:
        """Make Model callable by delegating to func."""
        # Merge model defaults with call-specific kwargs
        call_kwargs = {
            "temperature": self.temperature,
            **({"max_tokens": self.max_tokens} if self.max_tokens else {}),
            **kwargs
        }
        return self.func(messages, **call_kwargs)

@dataclass(frozen=True)
class Tool:
    """Immutable tool definition with rich metadata"""
    name: str
    description: str
    inputs: Dict[str, Dict[str, str]]
    output_type: str
    func: Callable
    output_schema: Optional[Dict[str, Any]] = None

    def __call__(self, *args, **kwargs) -> Any:
        """Make Tool callable"""
        return self.func(*args, **kwargs)

    def to_code_prompt(self) -> str:
        """Generate code prompt like original smolagents"""
        args_signature = ", ".join(
            f"{name}: {schema['type']}" for name, schema in self.inputs.items()
        )

        has_schema = self.output_schema is not None
        output_type = "dict" if has_schema else self.output_type
        tool_signature = f"({args_signature}) -> {output_type}"
        tool_doc = self.description

        if has_schema:
            tool_doc += "\n\nImportant: This tool returns structured output! Use the JSON schema below to directly access fields like result['field_name']. NO print() statements needed to inspect the output!"

        if self.inputs:
            args_descriptions = "\n".join(
                f"{name}: {schema['description']}" for name, schema in self.inputs.items()
            )
            args_doc = f"Args:\n{textwrap.indent(args_descriptions, '    ')}"
            tool_doc += f"\n\n{args_doc}"

        if has_schema:
            formatted_schema = json.dumps(self.output_schema, indent=4)
            indented_schema = textwrap.indent(formatted_schema, "        ")
            returns_doc = f"\nReturns:\n    dict (structured output): This tool ALWAYS returns a dictionary that strictly adheres to the following JSON schema:\n{indented_schema}"
            tool_doc += f"\n{returns_doc}"

        tool_doc = f'"""{tool_doc}\n"""'
        return f"def {self.name}{tool_signature}:\n{textwrap.indent(tool_doc, '    ')}"


@dataclass(frozen=True)
class CodeExecutionResult:
    """Result of code execution"""
    output: Any = None
    logs: str = ""
    is_final_answer: bool = False
    new_variables: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None

@dataclass(frozen=True)
class Executor:
    """A structured executor wrapper with metadata and configuration."""

    name: str
    func: Callable
    description: str = ""
    authorized_imports: List[str] = field(default_factory=lambda: ["math", "json", "re", "datetime", "random"])
    tools: Dict[str, Tool] = field(default_factory=dict)

    def __call__(self, code: str, variables: Dict[str, Any] = None) -> CodeExecutionResult:
        """Make Executor callable by delegating to func."""
        return self.func(code, variables or {})

@dataclass(frozen=True)
class ExecutionContext:
    """Context passed through execution pipeline"""
    config: AgentConfig
    state: AgentState
    model: Model
    tools: Dict[str, Tool]
    executor: Executor
    logger: Optional[Callable] = None

    def with_state(self, new_state: AgentState) -> 'ExecutionContext':
        """Update state - returns new context"""
        return replace(self, state=new_state)


@dataclass(frozen=True)
class StepResult:
    """Result of a single step"""
    output: Any
    is_final: bool
    observations: str
    new_variables: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None


# Action types
class Action(ABC):
    """Base action type"""
    pass


@dataclass(frozen=True)
class CodeAction(Action):
    """Code execution action"""
    code: str

@dataclass(frozen=True)
class ExecutionResult:
    """Result of action execution"""
    output: Any
    is_final: bool
    observations: str
    new_variables: Dict[str, Any]
    logs: str = ""


# Exception types
class AgentError(Exception):
    """Base agent error"""
    pass


class AgentGenerationError(AgentError):
    """Error in model generation"""
    pass


class AgentParsingError(AgentError):
    """Error in parsing model output"""
    pass


class AgentExecutionError(AgentError):
    """Error in execution"""
    pass


class AgentToolExecutionError(AgentError):
    """Error in tool execution"""
    pass


class AgentMaxStepsError(AgentError):
    """Max steps reached"""
    pass



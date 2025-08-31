#!/usr/bin/env python3
import re
from typing import List, Tuple, TypeVar, Sequence
from .core import AgentState, CodeAction, ExecutionContext, MemoryStep, ChatMessage, StepResult, SystemPromptStep, TaskStep, ReasoningStep, ToolCallingStep, ToolResponseStep, ToolCall

T = TypeVar('T')

def create_initial_state(task: str, system_prompt: str = "") -> AgentState:
    """Create initial agent state"""
    memory = []

    if system_prompt:
        memory.append(SystemPromptStep(system_prompt))

    memory.append(TaskStep(task))

    return AgentState(
        memory=memory,
        variables={},
        step_number=1,
        interrupted=False
    )

def generate_action(ctx: ExecutionContext) -> Tuple[ChatMessage, ExecutionContext]:
    """Generate next action from model"""
    messages = memory_to_messages(ctx.state.memory)
    response = ctx.model(messages)
    
    # Create and add ReasoningStep to memory
    reasoning_step = ReasoningStep(
        model_output=response.content,
        token_usage=response.token_usage
    )
    new_state = ctx.state.add_memory(reasoning_step)
    new_ctx = ctx.with_state(new_state)

    # TODO: these things basically redundant lol
    return response, new_ctx


def parse_action(response: ChatMessage, ctx: ExecutionContext) -> Tuple[CodeAction, ExecutionContext]:
    """Parse action from model output"""
    code = extract_code_from_content(response.content, ctx.config.code_block_tags)
    
    # Create ToolCall for python_interpreter (matches original smolagents)
    tool_call = ToolCall(
        name="python_interpreter",
        arguments=code,
        id=f"call_{len(ctx.state.memory)}"
    )
    
    # Create and add ToolCallingStep to memory
    tool_calling_step = ToolCallingStep(
        tool_calls=[tool_call],
        code_action=code
    )
    new_state = ctx.state.add_memory(tool_calling_step)
    new_ctx = ctx.with_state(new_state)

    # TODO: these things basically redundant lol
    return CodeAction(code=code), new_ctx

def execute_action(action: CodeAction, ctx: ExecutionContext) -> Tuple[StepResult, ExecutionContext]:
    """Execute code action"""
    # Create executor with tools injected if not already done
    # Execute code with current variables
    result = ctx.executor(action.code, ctx.state.variables)

    # Create observations
    observations = f"Execution logs:\n{result.logs}\nOutput: {result.output}"

    # Create and add ToolResponseStep to memory
    tool_response_step = ToolResponseStep(
        observations=observations,
        execution_output=result.output,
        new_variables=result.new_variables or {},
        error=result.error
    )

    # Update state with new variables and memory
    updated_state = ctx.state.update_variables(result.new_variables or {})
    updated_state = updated_state.add_memory(tool_response_step)
    new_ctx = ctx.with_state(updated_state)

    # TODO: this is basically redundant with all the stuff getting updated in ctx i
    # think - with the exception of the final_answer thing. so we should prob like just
    # add a is_final field to toolresponsestep and get rid of this
    return StepResult(
        output=result.output,
        is_final=result.is_final_answer,
        observations=observations,
        new_variables=result.new_variables or {}
    ), new_ctx

# Helpers

def memory_to_messages(memory: Sequence[MemoryStep]) -> List[ChatMessage]:
    """Convert memory steps to chat messages"""
    return flatten([mem.to_messages() for mem in memory])

def flatten(nested_list: List[List[T]]) -> List[T]:
    """Flatten a nested list"""
    return [item for sublist in nested_list for item in sublist]

def extract_code_from_content(content: str, code_block_tags: tuple) -> str:
    """Extract code from content using block tags"""
    opening_tag, closing_tag = code_block_tags
    pattern = f"{re.escape(opening_tag)}(.*?){re.escape(closing_tag)}"
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return last code block
    else:
        return ""

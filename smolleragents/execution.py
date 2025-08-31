#!/usr/bin/env python3

"""
Execution functions for code and tools
"""

import ast
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Any, Callable, Optional
from .core import AgentExecutionError, Executor, CodeExecutionResult, Tool
from .utils import map_parallel, substitute_variables


def create_safe_globals(authorized_imports: List[str]) -> Dict[str, Any]:
    """Create safe global environment for code execution"""
    import math
    import json
    import re
    import datetime
    import random

    safe_builtins = {
        'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
        'chr': chr, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
        'filter': filter, 'float': float, 'hex': hex, 'int': int, 'len': len,
        'list': list, 'map': map, 'max': max, 'min': min, 'oct': oct,
        'ord': ord, 'pow': pow, 'range': range, 'reversed': reversed,
        'round': round, 'set': set, 'sorted': sorted, 'str': str,
        'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
        'print': print,  # Allow print for output
    }

    # Add authorized imports
    safe_modules = {}
    available_modules = {
        'math': math, 'json': json, 're': re, 'datetime': datetime, 'random': random
    }

    if "*" in authorized_imports:
        safe_modules.update(available_modules)
    else:
        for module_name in authorized_imports:
            if module_name in available_modules:
                safe_modules[module_name] = available_modules[module_name]

    return {**safe_builtins, **safe_modules}


def validate_ast(tree: ast.AST, authorized_imports: List[str]):
    """Validate AST for security issues"""
    forbidden_nodes = [
        ast.Import, ast.ImportFrom,  # Imports handled separately
    ]

    for node in ast.walk(tree):
        # Check for forbidden constructs
        if any(isinstance(node, forbidden_type) for forbidden_type in forbidden_nodes):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Check if import is authorized
                if isinstance(node, ast.Import):
                    module_names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module_names = [node.module] if node.module else []

                if not ("*" in authorized_imports or all(name in authorized_imports for name in module_names)):
                    raise AgentExecutionError(f"Unauthorized import: {module_names}")

        # Check for dangerous attribute access
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise AgentExecutionError(f"Access to private attribute '{node.attr}' not allowed")


def check_for_final_answer(logs: str, locals_dict: Dict[str, Any]) -> bool:
    """Check if execution indicates a final answer"""
    # First check if final_answer() was actually called via global flag
    import builtins
    if hasattr(builtins, '_FINAL_ANSWER_TRIGGERED') and getattr(builtins, '_FINAL_ANSWER_TRIGGERED', False):
        # Reset the flag for next execution
        setattr(builtins, '_FINAL_ANSWER_TRIGGERED', False)
        return True
    
    # Fallback: check for text markers in logs
    final_markers = ["FINAL_ANSWER:", "final_answer:", "Final Answer:"]
    logs_lower = logs.lower()
    if any(marker.lower() in logs_lower for marker in final_markers):
        return True

    # Fallback: check for final_answer variable
    if 'final_answer' in locals_dict:
        return True

    return False


# TODO: yeah, let's prob just use inheritance here (and for tools, and for models.)
def create_local_python_executor(
    authorized_imports: List[str],
    name: str = "local-python",
    tools: Dict[str, Tool] = None
) -> Executor:
    """Create a local Python executor wrapper."""
    # Create safe execution environment
    safe_globals = create_safe_globals(authorized_imports)
    
    # TODO: hmm this should prob happen in the Executor __call__ or something.
    # Inject tools into safe_globals
    if tools:
        for tool_name, tool in tools.items():
            safe_globals[tool_name] = tool
    
    execution_locals = {}

    def executor_func(code: str, variables: Dict[str, Any] = None) -> CodeExecutionResult:
        """Execute Python code in safe environment"""
        nonlocal execution_locals

        # Track what existed before this execution
        existing_locals = execution_locals.copy()

        # Update locals with provided variables
        if variables:
            execution_locals.update(variables)

        # Create execution environment with proper globals/locals separation
        exec_locals = execution_locals.copy()

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Parse and validate code
            parsed = ast.parse(code)
            validate_ast(parsed, authorized_imports)

            # Execute with proper globals/locals separation
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compile(parsed, '<string>', 'exec'), safe_globals, exec_locals)

            # Get captured output
            logs = stdout_capture.getvalue()

            # Update persistent execution_locals with new/changed variables
            execution_locals.update(exec_locals)

            # Check for final answer markers
            is_final = check_for_final_answer(logs, exec_locals)

            # Get the last expression result if available
            output = exec_locals.get('_', None)
            if output is None and logs.strip():
                output = logs.strip().split('\n')[-1]

            # Extract only newly created or modified variables
            new_variables = {}
            for k, v in exec_locals.items():
                if (not k.startswith('_') and 
                    k not in safe_globals and 
                    (k not in existing_locals or existing_locals[k] != v)):
                    new_variables[k] = v

            return CodeExecutionResult(
                output=output,
                logs=logs,
                is_final_answer=is_final,
                new_variables=new_variables
            )

        except Exception as e:
            error_logs = stderr_capture.getvalue()
            if not error_logs:
                error_logs = traceback.format_exc()

            return CodeExecutionResult(
                output=None,
                logs=f"{stdout_capture.getvalue()}\nERROR: {error_logs}",
                error=e
            )

    return Executor(
        name=name,
        func=executor_func,
        description=f"Local Python executor with imports: {authorized_imports}",
        authorized_imports=authorized_imports,
        tools=tools
    )

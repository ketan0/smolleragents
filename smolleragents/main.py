import yaml
import argparse
from pathlib import Path
from .core import AgentConfig, ExecutionContext
from .execution import create_local_python_executor
from .pipeline import generate_action, memory_to_messages, parse_action, create_initial_state, execute_action#, update_memory
from .tools import create_basic_tools
from .models import create_openai_model
from jinja2 import Template

def main():
    """Main entry point for the smolleragents CLI."""
    parser = argparse.ArgumentParser(description="Run a smolleragent task")
    parser.add_argument("task", help="The task for the agent to perform")
    args = parser.parse_args()
    
    task = args.task
    config = AgentConfig()
    tools = create_basic_tools()

    # TODO: move this to helper
    with open(Path(__file__).parent / "code_agent.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    system_prompt_template = prompts["system_prompt"]
    opening_tag, closing_tag = config.code_block_tags
    template = Template(system_prompt_template)
    system_prompt = template.render(
        code_block_opening_tag=opening_tag,
        code_block_closing_tag=closing_tag,
        tools=tools,
        authorized_imports=config.authorized_imports
    )

    initial_state = create_initial_state(task, system_prompt)
    model = create_openai_model()
    executor = create_local_python_executor(config.authorized_imports, tools=tools)
    ctx = ExecutionContext(
        config=config,
        state=initial_state,
        model=model,
        tools=tools,
        executor=executor,
    )
    # print(ctx)
    # print(system_prompt)

    MAX_STEPS = 20000
    for _ in range(MAX_STEPS):
        response, ctx = generate_action(ctx)
        print(response.content)
        action, ctx = parse_action(response, ctx)
        result, ctx = execute_action(action, ctx)
        print(result.observations)
        if result.is_final:
            break


if __name__ == "__main__":
    main()

"""
Tool definitions and factories for functional smolagents
"""

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from .core import Tool

def create_final_answer_tool() -> Tool:
    """Create final answer tool"""
    def final_answer_impl(answer):
        # Use a global flag that can be detected by check_for_final_answer
        import builtins
        # Set a global flag that persists across the execution context
        setattr(builtins, '_FINAL_ANSWER_TRIGGERED', True)
        setattr(builtins, '_FINAL_ANSWER_VALUE', answer)
        print(f"FINAL_ANSWER: {answer}")  # Also print for additional detection
        return answer
    
    return Tool(
        name="final_answer",
        description="Provides the final answer to the given problem",
        inputs={"answer": {
            "type": "any",
            "description": "The final answer to the problem"
        }},
        output_type="any",
        func=final_answer_impl
    )


def create_web_search_tool() -> Tool:
    """Create web search tool using DuckDuckGo"""
    def web_search_impl(query: str) -> str:
        try:
            import requests
            from html.parser import HTMLParser
        except ImportError as e:
            raise ImportError("You must install 'requests' to run this tool: pip install requests") from e

        class SimpleResultParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = {}
                self.capture_title = False
                self.capture_description = False
                self.capture_link = False

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and attrs.get("class") == "result-link":
                    self.capture_title = True
                elif tag == "td" and attrs.get("class") == "result-snippet":
                    self.capture_description = True
                elif tag == "span" and attrs.get("class") == "link-text":
                    self.capture_link = True

            def handle_endtag(self, tag):
                if tag == "a" and self.capture_title:
                    self.capture_title = False
                elif tag == "td" and self.capture_description:
                    self.capture_description = False
                elif tag == "span" and self.capture_link:
                    self.capture_link = False
                elif tag == "tr":
                    if {"title", "description", "link"} <= self.current.keys():
                        self.current["description"] = " ".join(self.current["description"])
                        self.results.append(self.current)
                        self.current = {}

            def handle_data(self, data):
                if self.capture_title:
                    self.current["title"] = data.strip()
                elif self.capture_description:
                    self.current.setdefault("description", [])
                    self.current["description"].append(data.strip())
                elif self.capture_link:
                    self.current["link"] = "https://" + data.strip()

        try:
            response = requests.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            response.raise_for_status()

            parser = SimpleResultParser()
            parser.feed(response.text)
            results = parser.results[:10]  # Limit to 10 results

            if not results:
                return "No results found! Try a less restrictive/shorter query."

            formatted_results = []
            for result in results:
                formatted_results.append(f"[{result['title']}]({result['link']})\n{result['description']}")

            return "## Search Results\n\n" + "\n\n".join(formatted_results)

        except Exception as e:
            return f"Error performing web search: {str(e)}"

    return Tool(
        name="web_search",
        description="Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions",
        inputs={"query": {
            "type": "string",
            "description": "The search query to perform"
        }},
        output_type="string",
        func=web_search_impl
    )


def create_visit_webpage_tool(max_output_length: int = 40000) -> Tool:
    """Create webpage visit tool"""
    def visit_webpage_impl(url: str) -> str:
        try:
            import re
            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
        except ImportError as e:
            raise ImportError("You must install packages 'markdownify' and 'requests': pip install markdownify requests") from e

        def truncate_content(content: str, max_length: int) -> str:
            if len(content) <= max_length:
                return content
            return content[:max_length] + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"

        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()

            # Convert HTML to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return truncate_content(markdown_content, max_output_length)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    return Tool(
        name="visit_webpage",
        description="Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages",
        inputs={"url": {
            "type": "string",
            "description": "The url of the webpage to visit"
        }},
        output_type="string",
        func=visit_webpage_impl
    )

def create_basic_tools() -> Dict[str, Tool]:
    """Create basic tool set"""
    return {
        "web_search": create_web_search_tool(),
        "visit_webpage": create_visit_webpage_tool(),
        "final_answer": create_final_answer_tool(),
    }

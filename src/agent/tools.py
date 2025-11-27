"""
Definizioni dei Tool per Function Calling.

Questo modulo contiene le definizioni JSON Schema dei tool che il modello
può usare per interagire con sistemi esterni (RAG, API, etc.).

I tool seguono il formato OpenAI Function Calling:
- name: Nome della funzione
- description: Descrizione per il modello
- parameters: JSON Schema dei parametri

Questi vengono usati per:
1. Generare il prompt con i tool disponibili
2. Validare le chiamate del modello
3. Eseguire le funzioni reali
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definisce un parametro di un tool."""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None


@dataclass
class Tool:
    """Definisce un tool per function calling."""
    
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None  # Funzione da eseguire
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Converte il tool in JSON Schema per OpenAI/ChatML format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    
    def to_prompt_format(self) -> str:
        """Genera una descrizione leggibile per il prompt."""
        params_str = ", ".join(
            f"{p.name}: {p.type}" + (f" = {p.default}" if p.default else "")
            for p in self.parameters
        )
        return f"{self.name}({params_str}) - {self.description}"


# =============================================================================
# TOOL DEFINITIONS - RAG / Knowledge Base
# =============================================================================

SEARCH_KNOWLEDGE_BASE = Tool(
    name="search_knowledge_base",
    description=(
        "Search the knowledge base for relevant information. "
        "Use this tool when you need to find specific facts, documentation, "
        "or context that may help answer the user's question. "
        "Returns the most relevant documents from the knowledge base."
    ),
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="The search query to find relevant information in the knowledge base.",
            required=True,
        ),
        ToolParameter(
            name="n_results",
            type="integer",
            description="Number of results to return (default: 3, max: 10).",
            required=False,
            default=3,
        ),
        ToolParameter(
            name="filter_source",
            type="string",
            description="Optional filter by source/category (e.g., 'documentation', 'code', 'faq').",
            required=False,
        ),
    ],
)


# =============================================================================
# TOOL DEFINITIONS - Code Execution
# =============================================================================

EXECUTE_PYTHON = Tool(
    name="execute_python",
    description=(
        "Execute Python code in a sandboxed environment. "
        "Use this to run calculations, data processing, or test code snippets. "
        "Returns the output or any errors from execution."
    ),
    parameters=[
        ToolParameter(
            name="code",
            type="string",
            description="The Python code to execute.",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Maximum execution time in seconds (default: 30).",
            required=False,
            default=30,
        ),
    ],
)


# =============================================================================
# TOOL DEFINITIONS - Web / API
# =============================================================================

WEB_SEARCH = Tool(
    name="web_search",
    description=(
        "Search the web for current information. "
        "Use this when the user asks about recent events, news, "
        "or information that may not be in your training data."
    ),
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="The search query.",
            required=True,
        ),
        ToolParameter(
            name="num_results",
            type="integer",
            description="Number of results to return (default: 5).",
            required=False,
            default=5,
        ),
    ],
)


GET_WEATHER = Tool(
    name="get_weather",
    description="Get the current weather for a specified location.",
    parameters=[
        ToolParameter(
            name="location",
            type="string",
            description="The city and country (e.g., 'Rome, Italy').",
            required=True,
        ),
        ToolParameter(
            name="units",
            type="string",
            description="Temperature units.",
            required=False,
            enum=["celsius", "fahrenheit"],
            default="celsius",
        ),
    ],
)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

# Tutti i tool disponibili
ALL_TOOLS: Dict[str, Tool] = {
    "search_knowledge_base": SEARCH_KNOWLEDGE_BASE,
    "execute_python": EXECUTE_PYTHON,
    "web_search": WEB_SEARCH,
    "get_weather": GET_WEATHER,
}

# Tool di default per training RAG/Agentic
DEFAULT_RAG_TOOLS: List[Tool] = [
    SEARCH_KNOWLEDGE_BASE,
]

DEFAULT_AGENT_TOOLS: List[Tool] = [
    SEARCH_KNOWLEDGE_BASE,
    EXECUTE_PYTHON,
    WEB_SEARCH,
]


def get_tools_json_schema(tools: List[Tool]) -> List[Dict[str, Any]]:
    """
    Converte una lista di tool in JSON Schema.
    
    Args:
        tools: Lista di Tool
        
    Returns:
        Lista di JSON Schema per OpenAI format
    """
    return [tool.to_json_schema() for tool in tools]


def get_tools_prompt(tools: List[Tool]) -> str:
    """
    Genera una descrizione dei tool per il prompt.
    
    Args:
        tools: Lista di Tool
        
    Returns:
        Stringa formattata con la descrizione dei tool
    """
    lines = ["Available tools:"]
    for tool in tools:
        lines.append(f"- {tool.to_prompt_format()}")
    return "\n".join(lines)


def format_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """
    Formatta una chiamata a tool nel formato ChatML/function_call.
    
    Args:
        name: Nome del tool
        arguments: Argomenti della chiamata
        
    Returns:
        Stringa JSON formattata
    """
    return json.dumps({
        "name": name,
        "arguments": arguments,
    }, indent=2)


def parse_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """
    Parsa una risposta del modello per estrarre la chiamata a tool.
    
    Supporta vari formati:
    - <function_call>{"name": ..., "arguments": ...}</function_call>
    - ```json\n{"name": ..., "arguments": ...}\n```
    - {"name": ..., "arguments": ...}
    
    Args:
        response: Risposta del modello
        
    Returns:
        Dict con 'name' e 'arguments', o None se non trovato
    """
    import re
    
    # Pattern per estrarre la chiamata
    patterns = [
        r'<function_call>\s*(\{.*?\})\s*</function_call>',
        r'```(?:json)?\s*(\{.*?\})\s*```',
        r'(\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if "name" in data:
                    return data
            except json.JSONDecodeError:
                continue
    
    return None


# =============================================================================
# SYSTEM PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT_WITH_TOOLS = """You are a helpful AI assistant with access to tools.

{tools_description}

When you need to use a tool, respond with a function call in this exact format:
<function_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
</function_call>

After receiving the tool result, provide your final answer to the user.
If you don't need any tools, respond directly to the user."""


SYSTEM_PROMPT_RAG = """You are a helpful AI assistant with access to a knowledge base.

When answering questions, you should:
1. First search the knowledge base for relevant information
2. Use the retrieved context to formulate your answer
3. If the answer is not in the context, say so clearly
4. Always cite your sources when using information from the knowledge base

Available tool:
- search_knowledge_base(query: str) - Search for relevant information

To search, use:
<function_call>
{{"name": "search_knowledge_base", "arguments": {{"query": "your search query"}}}}
</function_call>"""


def get_system_prompt_with_tools(tools: List[Tool]) -> str:
    """Genera il system prompt con i tool specificati."""
    tools_desc = get_tools_prompt(tools)
    return SYSTEM_PROMPT_WITH_TOOLS.format(tools_description=tools_desc)


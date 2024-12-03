import os
import json
import logging

import config

import requests
import difflib
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.agent_toolkits.openapi.spec import (
    reduce_openapi_spec,
    ReducedOpenAPISpec,
)
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description
from infrahub_sdk import Config, InfrahubClientSync
import urllib3

# Configure logging
logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global variables for lazy initialization
llm = None
agent_executor = None


logger = logging.getLogger(__name__)


# InfrahubController for CRUD Operations
class InfrahubController:
    def __init__(self, infrahub_url, api_token):
        self.infrahub = infrahub_url.rstrip("/")
        self.api_token = api_token
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }
        config = Config(
            echo_graphql_queries=True, api_token=api_token, address=infrahub_url
        )
        self.client = InfrahubClientSync(config=config)

    def get_api(self, api_url: str, params: dict = None):
        response = requests.get(
            f"{self.infrahub}{api_url}",
            headers=self.headers,
            params=params,
            verify=False,
        )
        response.raise_for_status()
        return response.json()

    def post_api(self, api_url: str, payload: dict):
        response = requests.post(
            f"{self.infrahub}{api_url}",
            headers=self.headers,
            json=payload,
            verify=False,
        )
        response.raise_for_status()
        return response.json()

    def delete_api(self, api_url: str):
        response = requests.delete(
            f"{self.infrahub}{api_url}", headers=self.headers, verify=False
        )
        response.raise_for_status()
        return response.json()

    def graphql(self, query: str, at: str):
        data = self.client.execute_graphql(query=query, at=at)
        return data


def _load_openapi(file_path=config.open_api_file):
    with open(file_path, "r") as f:
        schema = json.load(f)
        schema["servers"] = {"url": "http://asdf"}  # TODO
        data = reduce_openapi_spec(schema)
    return data


# Function to load supported URLs with their names from a JSON file
def load_urls(file_path=config.open_api_file):
    if not os.path.exists(file_path):
        logger.warning(f"URLs file '{file_path}' not found.")
        return {"error": f"URLs file '{file_path}' not found."}
    try:
        return _load_openapi(file_path)
    except Exception as e:
        logger.warning(f"Error loading URLs: {str(e)}")
        return {"error": f"Error loading URLs: {str(e)}"}


def check_url_support(api_url: str) -> dict:
    url_list = load_urls()
    # split operation and route and only take route/url
    if isinstance(url_list, ReducedOpenAPISpec):
        urls = [route.split()[-1] for route, _, _ in url_list.endpoints]

        close_url_matches = difflib.get_close_matches(api_url, urls, n=1, cutoff=0.6)

        if close_url_matches:
            closest_url = close_url_matches[0]
            return {
                "status": "supported",
                "closest_url": closest_url,
            }
    logger.warning(f"The input '{api_url}' is not supported.")
    return {
        "status": "unsupported",
        "message": f"The input '{api_url}' is not supported.",
    }


# Tools for interacting with infrahub
@tool
def discover_apis(dummy_input: str = None) -> dict:
    """Discover available infrahub APIs from a local JSON file."""
    try:
        if not os.path.exists(config.open_api_file):
            return {
                "error": "API JSON file not found. Please ensure 'infrahub_apis.json' exists in the project directory."
            }
        with open(config.open_api_file) as fp:
            schema = json.load(fp)
        urls = []
        for path, methods in schema.get("paths", {}).items():
            for method, definition in methods.items():
                urls.append((method, path, definition.get("summary")))
        return {"apis": urls, "message": "APIs successfully loaded from JSON file"}
    except Exception as e:
        return {"error": f"An error occurred while loading the APIs: {str(e)}"}


@tool
def check_supported_url_tool(api_url: str) -> dict:
    """Check if an API URL or Name is supported by infrahub."""
    result = check_url_support(api_url)
    if result.get("status") == "supported":
        closest_url = result["closest_url"]
        return {
            "status": "supported",
            "message": f"The closest supported API URL is '{closest_url}'.",
            "action": {"next_tool": "get_infrahub_data_tool", "input": closest_url},
        }
    return result


@tool
def get_infrahub_data_tool(api_url: str) -> dict:
    """Fetch data from Infrahub."""
    try:
        infrahub_controller = InfrahubController(
            infrahub_url=os.getenv("INFRAHUB_URL"),
            api_token=os.getenv("INFRAHUB_TOKEN"),
        )
        data = infrahub_controller.get_api(api_url)
        return data
    except requests.HTTPError as e:
        return {"error": f"Failed to fetch data from Infrahub: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@tool
def create_infrahub_data_tool(input: str) -> dict:
    """Create new data in Infrahub."""
    try:
        data = json.loads(input)
        api_url = data.get("api_url")
        payload = data.get("payload")

        if not api_url or not payload:
            raise ValueError("Both 'api_url' and 'payload' must be provided.")

        if not isinstance(payload, dict):
            raise ValueError("Payload must be a dictionary.")

        infrahub_controller = InfrahubController(
            infrahub_url=os.getenv("INFRAHUB_URL"),
            api_token=os.getenv("INFRAHUB_TOKEN"),
        )
        return infrahub_controller.post_api(api_url, payload)
    except Exception as e:
        return {"error": f"An error occurred in create_infrahub_data_tool: {str(e)}"}


@tool
def delete_infrahub_data_tool(api_url: str) -> dict:
    """Delete data from Infrahub."""
    try:
        infrahub_controller = InfrahubController(
            infrahub_url=os.getenv("INFRAHUB_URL"),
            api_token=os.getenv("INFRAHUB_TOKEN"),
        )
        return infrahub_controller.delete_api(api_url)
    except requests.HTTPError as e:
        return {"error": f"Failed to delete data from Infrahub: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@tool
def querry_infrahub_data(query: str, timestamp: str = None) -> dict:
    """Use GraphQL to querry data from infrahub. Use at to specify a Timestamp"""

    # Need a better way to validate query syntax
    if "`" in query:
        return {"error": f"Do **not** wrap GraphQL queries or mutations in Markdown-style code blocks."}

    try:
        infrahub_controller = InfrahubController(
            infrahub_url=os.getenv("INFRAHUB_URL"),
            api_token=os.getenv("INFRAHUB_TOKEN"),
        )
        return infrahub_controller.graphql(query, timestamp)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


def process_agent_response(response):
    if (
        response
        and response.get("status") == "supported"
        and "next_tool" in response.get("action", {})
    ):
        next_tool = response["action"]["next_tool"]
        tool_input = response["action"]["input"]

        # Automatically invoke the next tool
        return agent_executor.invoke(
            {
                "input": tool_input,
                "chat_history": st.session_state.chat_history,
                "agent_scratchpad": "",
                "tool": next_tool,
            }
        )
    else:
        return response


# ============================================================
# Streamlit App
# ============================================================


def configure_page():
    st.title("🦦 Infrahub 🦦 Configuration")
    base_url = st.text_input("Infrahub URL", placeholder="http://localhost:8000")
    api_token = st.text_input(
        "Infrahub API Token", type="password", placeholder="Your API Token"
    )
    openai_key = st.text_input(
        "OpenAI API Key", type="password", placeholder="Your OpenAI API Key"
    )

    if st.button("Save and Continue"):
        if not base_url or not api_token or not openai_key:
            st.error("All fields are required.")
        else:
            st.session_state["INFRAHUB_URL"] = base_url
            st.session_state["INFRAHUB_TOKEN"] = api_token
            st.session_state["OPENAI_API_KEY"] = openai_key
            os.environ["INFRAHUB_URL"] = base_url
            os.environ["INFRAHUB_TOKEN"] = api_token
            os.environ["OPENAI_API_KEY"] = openai_key
            st.success("Configuration saved! Redirecting to chat...")
            st.session_state["page"] = "chat"


def initialize_agent():
    global llm, agent_executor
    if not llm:
        # Initialize the LLM with the API key from session state
        llm = ChatOpenAI(
            model_name="gpt-4o", openai_api_key=st.session_state["OPENAI_API_KEY"]
        )

        # Define tools
        tools = [
            discover_apis,
            check_supported_url_tool,
            get_infrahub_data_tool,
            create_infrahub_data_tool,
            delete_infrahub_data_tool,
            querry_infrahub_data,
        ]

        # Create the prompt template
        tool_descriptions = render_text_description(tools)
        # Create the PromptTemplate
        template = """
Assistant is a network assistant capable of managing Infrahub data using CRUD operations.
Infrahub is a temporal graph database built on three key pillars:
a flexible schema, version control, and unified storage. 
It primarily uses GraphQL for interactions, complemented by REST APIs for specific workflows.
Queries and mutations in GraphQL are tailored to the schema, offering granular control over data retrieval and modifications.

TOOLS:
- discover_apis: Discovers available Infrahub APIs from a local JSON file.
- check_supported_url_tool: Checks if an API URL or Name is supported by Infrahub.
- get_infrahub_data_tool: Fetches data from Infrahub using the specified API URL.
- create_infrahub_data_tool: Creates new data in Infrahub using the specified API URL and payload.
- delete_infrahub_data_tool: Deletes data from Infrahub using the specified API URL.
- querry_infrahub_data: Queries data from Infrahub using GraphQL. Accepts a query in string format and an optional timestamp (parsed with the Pendulum library) to retrieve historical data.

GUIDELINES:
1. **Response Format**:
   - Always follow the structured format strictly.
   - If a `Final Answer` is required, ensure that all intermediate steps (`Thought`, `Action`, `Action Input`, and `Observation`) precede it.
   - Do not skip fields or combine `Observation` and `Final Answer`.

2. **GraphQL Querying**:
   - GraphQL is the primary interface for interacting with Infrahub. Use queries for data retrieval and mutations for data modifications (create, update, upsert, delete).
   - Queries follow a nested structure (e.g., `edges > node`) to support pagination and detailed attributes access.
   - Mutations require input under the `data` field, with `id` or `hfid` for updates and deletes.
   - Ensure responses include only requested fields to minimize overhead.
   - Do **not** wrap GraphQL queries or mutations in Markdown-style code blocks.
   - The following query will return the name of all the devices in the database: `query {{ InfraDevice {{ edges {{ node {{ name {{ value }} }} }} }} }}`
   - Use the schema summary REST APIs to identify the names of defined kinds in the schema. Avoid exhaustive schema retrieval.

3. **Schema and Relationships**:
   - Attributes and relationships (one-to-one or one-to-many) have distinct query formats. Be explicit when accessing metadata, properties, or nested relationships.
   - Automatically available fields for all nodes include `id`, `hfid`, and `display_label`.
   - Use relationships effectively to navigate the graph and establish connections between models.
   - Query all interfaces and IP addresses for ord1-edge.
    ```
    query DeviceIPAddresses {{
    InfraInterfaceL3(device__name__value:"ord1-edge1") {{
        edges {{
        node {{
            name {{ value }}
            description {{ value }}
            ip_addresses {{
            edges {{
                node {{
                address {{
                    value
                }}
                }}
            }}
            }}
        }}
        }}
    }}
    }}
    ```

4. **Temporal Data**:
   - Infrahub supports temporal queries. When using `querry_infrahub_data`, optionally specify a timestamp for historical data retrieval. Use Pendulum to parse timestamps accurately.

5. **Tool Usage**:
   - Use `discover_apis` to find API URLs if needed.
   - Use `check_supported_url_tool` to validate ambiguous URLs.
   - If certain about the URL or schema, directly use the corresponding CRUD tools or GraphQL queries/mutations.
   - Avoid collecting the entire schema; focus on summaries and specific queries to maintain efficiency.


FORMAT:
Thought: [Your thought process]
Action: [Tool Name]
Action Input: [Tool Input]
Observation: [Tool Response]
Final Answer: [Your response to the user]

BEGIN:

Previous conversation history:
{chat_history}

New input: {input}

{agent_scratchpad}
"""
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": tool_descriptions,
                "tool_names": ", ".join([t.name for t in tools]),
            },
        )

        # Create the ReAct agent
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

        # Create the AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=10,
        )


def chat_page():
    st.title("Chat with 🦦 Infrahub 🦦 AI Agent")
    user_input = st.text_input("Ask otto a question:", key="user_input")

    # Ensure the agent is initialized
    if "OPENAI_API_KEY" not in st.session_state:
        st.error("Please configure Infrahub and OpenAI settings first!")
        st.session_state["page"] = "configure"
        return

    initialize_agent()

    # Initialize session state variables if not already set
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Button to submit the question
    if st.button("Send"):
        if user_input:
            # Add the user input to the conversation history
            st.session_state.conversation.append(
                {"role": "user", "content": user_input}
            )

            # Invoke the agent with the user input and current chat history
            try:
                response = agent_executor.invoke(
                    {
                        "input": user_input,
                        "chat_history": st.session_state.chat_history,
                        "agent_scratchpad": "",  # Initialize agent scratchpad as an empty string
                    }
                )

                # Process the agent's response
                final_response = process_agent_response(response)

                # Extract the final answer
                final_answer = final_response.get("output", "No answer provided.")

                # Display the question and answer
                st.write(f"**Question:** {user_input}")
                st.write(f"**Answer:** {final_answer}")

                # Add the response to the conversation history
                st.session_state.conversation.append(
                    {"role": "assistant", "content": final_answer}
                )

                # Update chat history with the new conversation
                st.session_state.chat_history = "\n".join(
                    [
                        f"{entry['role'].capitalize()}: {entry['content']}"
                        for entry in st.session_state.conversation
                    ]
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display conversation history
    if st.session_state.conversation:
        st.markdown("### Conversation History")
        for entry in st.session_state.conversation:
            if entry["role"] == "user":
                st.markdown(f"**User:** {entry['content']}")
            elif entry["role"] == "assistant":
                st.markdown(f"**Infrahub AI ReAct Agent:** {entry['content']}")


if __name__ == "__main__":
    # Page Navigation
    if "page" not in st.session_state:
        st.session_state["page"] = "configure"

    if st.session_state["page"] == "configure":
        configure_page()
    elif st.session_state["page"] == "chat":
        chat_page()

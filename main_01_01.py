import asyncio
import operator
import os
from collections.abc import Sequence
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

load_dotenv()

UK_DESTINATIONS = [
    "Cornwall",
    "North_Cornwall",
    "South_Cornwall",
    "West_Cornwall",
]


async def build_vectorstore(destinations: Sequence[str]) -> Chroma:  # B
    """Download WikiVoyage pages and create
    a Chroma vector store."""
    urls = [f"https://en.wikivoyage.org/wiki/{slug}" for slug in destinations]  # C
    loader = AsyncHtmlLoader(urls)  # C
    print("Downloading destination pages ...")  # C
    docs = await loader.aload()  # C

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)  # D
    chunks = splitter.split_documents(docs)  # D

    print(f"Embedding {len(chunks)} chunks ...")  # E
    vectordb_client = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())  # E
    print("Vector store ready.\n")
    return vectordb_client  # F


# Singleton pattern (build once)
_ti_vectorstore_client: Chroma | None = None  # G


def get_travel_info_vectorstore() -> Chroma:  # H
    global _ti_vectorstore_client
    if _ti_vectorstore_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                """Set the OPENAI_API_KEY env 
                variable and re-run."""
            )
        _ti_vectorstore_client = asyncio.run(build_vectorstore(UK_DESTINATIONS))
    return _ti_vectorstore_client  # I


ti_vectorstore_client = get_travel_info_vectorstore()  # J
ti_retriever = ti_vectorstore_client.as_retriever()  # K

# A Destination list; you can add more destinations here
# B Function to build the vectorstore and return a reference to the vectorstore client
# C Load the destination pages asynchronously from the web into a list of documents
# D Split the documents into chunks of 1024 characters with 128 characters of overlap
# E Embed the chunks and store them in the vectorstore
# F Return the vectorstore client
# G Initialize a cache for the vectorstore client instance as None
# H Function to trigger the creation of the vectorstore and return a reference to the cache of its client instance
# I Return the a reference to the cache of the vectorstore client instance
# J Instantiate the vectorstore client
# K Instantiate the vectorstore retriever

# INFO:  ----------------------------------------------------------------------------
# INFO:  2. Define the only tool
# INFO:  ----------------------------------------------------------------------------


@tool  # A
def search_travel_info(query: str) -> str:  # B
    """Search embedded WikiVoyage content for
    information about destinations in England."""
    docs = ti_retriever.invoke(query)  # C
    top = docs[:4] if isinstance(docs, list) else docs  # C
    return "\n---\n".join(d.page_content for d in top)  # D


# A Define the tool using the @tool decorator
# B Define the tool function, which takes a query, performs a semantic search
#   and returns a string response from the vectorstore
# C Perform a semantic search on the vectorstore and return the top 4 results
# D Joins the top 4 results into a single string

# ----------------------------------------------------------------------------
# 3. Configure LLM with tool awareness
# ----------------------------------------------------------------------------
TOOLS = [search_travel_info]  # A

llm_model = ChatOpenAI(
    model="gpt-5-mini",  # B
    use_responses_api=True,
)  # B
llm_with_tools = llm_model.bind_tools(TOOLS)  # C

# A Define the tools list (in our case, only one tool)
# B Instantiate the LLM model with the gpt-5-mini model and the responses API
# C Bind the tools to the LLM model, which will generate a response with the tool calls

# ----------------------------------------------------------------------------
# 4. Initialize the dependencies for the LangGraph graph
# ----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# AgentState: it only contains LLM messages
# -----------------------------------------------------------------------------
class AgentState(TypedDict):  # A
    messages: Annotated[Sequence[BaseMessage], operator.add]  # B


# A Define the agent state
# B The agent state only contains LLM messages, which are appended to the list of messages

# -----------------------------------------------------------------------------
# CustomToolNode
# -----------------------------------------------------------------------------


class ToolsExecutionNode:  # A
    """Execute tools requested by the LLM in the last AIMessage."""

    def __init__(self, tools: Sequence):  # B
        self._tools_by_name = {t.name: t for t in tools}

    def __call__(self, state: dict):  # C
        messages: Sequence[BaseMessage] = state.get("messages", [])

        last_msg = messages[-1]  # D
        tool_messages: list[ToolMessage] = []  # E
        tool_calls = getattr(last_msg, "tool_calls", [])  # F

        for tool_call in tool_calls:  # G
            tool_name = tool_call["name"]  # H
            tool_args = tool_call["args"]  # I
            tool = self._tools_by_name[tool_name]  # J
            result = tool.invoke(tool_args)  # K
            tool_messages.append(
                ToolMessage(
                    content=result,  # L
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": tool_messages}  # M


tools_execution_node = ToolsExecutionNode(TOOLS)  # N

# A Define the tools execution node
# B Initialize the tools execution node with the tools list
# C Define the __call__ method, which is called when the node is invoked
# D Get the last message from the messages list
# E Initialize the tool messages list, to gather the results of the tool calls
# F Get the tool calls from the last message
# G Iterate over the tool calls
# H Get the tool name from the tool call
# I Get the tool arguments from the tool call
# J Get the tool from the tools list
# K Invoke the tool with the arguments
# L Add the tool result to the tool messages list
# M Return the tool messages list, which contains the results of the tool calls
# N Instantiate the tools execution node, to be used as a node in the LangGraph graph


# ----------------------------------------------------------------------------
# LLM node
# ----------------------------------------------------------------------------


def llm_node(state: AgentState):  # A
    """LLM node that decides whether to call the search tool."""
    current_messages = state["messages"]  # B
    response_message = llm_with_tools.invoke(current_messages)  # C

    return {"messages": [response_message]}  # D


# A Define the LLM node
# B Get the current messages from the agent state
# C Invoke the LLM model with the current messages. The LLM will decide whether to call the search tool or return an answer.
# D Return the response message, which contains the tool call or the answer

# ----------------------------------------------------------------------------
# 4. Build the LangGraph graph (llm_node + CustomToolNode)
# ----------------------------------------------------------------------------

builder = StateGraph(AgentState)  # A
builder.add_node("llm_node", llm_node)  # B
builder.add_node("tools", tools_execution_node)  # B

builder.add_conditional_edges("llm_node", tools_condition)  # C

builder.add_edge("tools", "llm_node")  # D

builder.set_entry_point("llm_node")  # E
travel_info_agent = builder.compile()  # F

# A Define the graph builder
# B Add the LLM node and the tools node to the graph
# C Add the conditional edges to the graph, to decide whether to execute the tool calls or return an answer and exit the graph
# D Add the edge from the tools node to the LLM node
# E Set the entry point to the LLM node
# F Compile the graph

# ----------------------------------------------------------------------------
# 5. Simple CLI interface
# ----------------------------------------------------------------------------

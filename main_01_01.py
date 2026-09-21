import asyncio
import os
from collections.abc import Sequence

from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

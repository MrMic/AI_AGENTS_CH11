import asyncio
import os
from collections.abc import Sequence

from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
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

# INFO:  A Destination list; you can add more destinations here
# INFO:  B Function to build the vectorstore and return a reference to the vectorstore client
# INFO:  C Load the destination pages asynchronously from the web into a list of documents
# INFO:  D Split the documents into chunks of 1024 characters with 128 characters of overlap
# INFO:  E Embed the chunks and store them in the vectorstore
# INFO:  F Return the vectorstore client
# INFO:  G Initialize a cache for the vectorstore client instance as None
# INFO:  H Function to trigger the creation of the vectorstore and return a reference to the cache of its client instance
# INFO:  I Return the a reference to the cache of the vectorstore client instance
# INFO:  J Instantiate the vectorstore client
# INFO:  K Instantiate the vectorstore retriever

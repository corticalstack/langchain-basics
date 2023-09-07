import os
from dotenv import load_dotenv

load_dotenv()

from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

"""
Tools Setup

We initialize different tools that the AI agent can use to complete tasks. In our case, the tools 
are Search, WriteFileTool, and ReadFileTool. The Search tool utilizes a GoogleSearchAPIWrapper to 
fetch real-time information from the internet, which can be employed for questions about current 
events or queries that need up-to-date information. The WriteFileTool and ReadFileTool manage 
file-related tasks. These tools are collected into a list that will be later passed to the agent. 
"""

search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="Useful for when you need to answer questions about current events. You should ask targeted questions",
        return_direct=True
    ),
    WriteFileTool(),
    ReadFileTool(),
]

# Set up the memory
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
embedding_size = 1536

"""
Agent Memory Setup

For the memory, we create the FAISS vector DB (but any other vector DB would work similarly), an 
efficient similarity search, and clustering of dense vectors. This is paired with an InMemoryDocstore 
instance for storing documents in memory and an OpenAIEmbeddings model for creating embeddings of the 
queries. These tools are crucial for the agent's remembering and retrieving past interactions.

AutoGPT has been designed to operate over longer periods. AutoGPT has incorporated a retrieval-based 
memory system that functions over intermediate agent steps to do that. 

This memory performs a semantic search across embeddings using the vector DB. While such retrieval-based 
memory is a part of LangChain, it was traditionally used for user and agent interactions, not agent and 
tools. AutoGPT's new adaptation represents a significant shift in how this memory system is applied.
"""

import faiss
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


"""
Setting up the Model and AutoGPT

Here we initialize the AutoGPT agent, giving it a name ("Jim") and a role ("Assistant"). We also 
supplied it with the tools and memory systems that were established in the previous steps. The language 
model being used here is ChatOpenAI, which is set to have a temperature of 0 
(indicating deterministic responses).
"""
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI

agent = AutoGPT.from_llm_and_tools(
    ai_name="Jim",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    memory=vectorstore.as_retriever()
)

# Set verbose to be true
agent.chain.verbose = True


"""
Running an Example

Finally, we provided an example task for the AutoGPT agent. This task ("Provide an analysis of the major 
historical events that led to the French Revolution") is complex and requires the agent to utilize its 
tools and memory system effectively to generate a response.

The agent takes some minutes to generate the final answer, but we get a peek into all the intermediate 
computations thanks to having set the verbose variable to True.

Since there are a lot of intermediate computations and the output is very long, we’ll see here only 
its crucial parts, giving a quick explanation of them.
"""
task = "Provide an analysis of the major historical events that led to the French Revolution"

agent.run([task])





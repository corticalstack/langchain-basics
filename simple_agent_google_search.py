from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

load_dotenv()

"""
Agents in LangChain help decide which actions to take based on user input. 
The example demonstrates initializing and using a "zero-shot-react-description" 
agent with a Google search tool.
"""

llm = OpenAI(model="text-davinci-003", temperature=0)

# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API.
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="google-search",  # unique identifier for the tool
        func=search.run,
        description="useful for when you need to search google to answer questions about current events",
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # uses the ReAct framework to decide which tool to use based only on the tool's description.
    verbose=True,  # Detailed infromation about what the agent is doing
    max_iterations=6,
)  # Prevent infinite loops

response = agent("Was India able to successfully land at the South Pole of the Moon?")
print(response["output"])

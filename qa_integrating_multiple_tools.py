import os
from dotenv import load_dotenv

load_dotenv()

from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType

from langchain.utilities import GoogleSearchAPIWrapper, PythonREPL

"""
As developers and information enthusiasts, we often find ourselves needing to utilize various 
tools and libraries to fetch and process data. By leveraging multiple tools simultaneously, 
we can create powerful, efficient, and comprehensive solutions for the systems we build with 
LangChain.  This lesson will demonstrate a practical example of combining the power of Google Search 
with the versatile Python-REPL tool for an effective result. You will learn how to harness the potential 
of multiple tools working together to streamline your own information retrieval projects.

Lets be more specific about what exactly we want to accomplish:

1. Find the answer to a query by searching the web: The agent should use its tools and language model to identify the most relevant sources for it.
2. Save the answer to a file: After retrieving the answer, the agent is expected to save it to a text file.
"""

from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType

from langchain.utilities import GoogleSearchAPIWrapper, PythonREPL

search = GoogleSearchAPIWrapper()
python_repl = PythonREPL()

"""
Here we have our toolkit set assembled of:

1. The google-search tool is a convenient way to perform Google searches when an agent needs information about current events. The tool makes use of Google's API to provide relevant search results.
2. The python_repl tool: This tool wraps a Python shell, allowing the execution of Python commands directly.
"""

toolkit = [
    Tool(
        name="google-search",
        func=search.run,
        description="useful for when you need to search Google to answer questions about current events",
    ),
    Tool(
        name="python_repl",
        description="A Python shell. Use this to execute Python commands. Input should be a valid Python command. Useful for saving strings to files.",
        func=python_repl.run,
    ),
]

"""
These tools are then added to the toolkit list, which is used to initialize an agent with the specified tools. 
The agent can then perform various tasks using the tools in its toolkit. The agent can be easily extended by 
adding more tools to the toolkit, allowing it to handle a wide range of tasks and situations. Let’s instantiate the agent.
"""
llm = OpenAI(model="text-davinci-003", temperature=0)

agent = initialize_agent(
    toolkit, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

"""
The parameter agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION specifies the agent's strategy, which means that the agent 
will attempt to perform tasks without any prior examples, relying solely on its understanding of the problem description 
and the available tools (and their descriptions).

Now let’s run the experiment! We should be able to ask the Agent directly by giving him instructions on what we want:
"""
agent.run(
    "Find the birth date of Napoleon Bonaparte and save it to a file 'answer.txt'."
)

"""
As you can see from the printed output, the agent first used the google-search tool with the query "Napoleon Bonaparte birth date". 
Upon seeing its result, the agent then wrote the following Python program to save the answer to the answer.txt local file:
"""
query = (
    "Find when Napoleon Bonaparte died and append this information "
    "to the content of the 'answer.txt' file in a new line."
)

agent.run(query)

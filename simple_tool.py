import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

load_dotenv()

"""
Tools in LangChain
LangChain provides a variety of tools for agents to interact with the outside world. 
These tools can be used to create custom agents that perform various tasks, such as
searching the web, answering questions, or running Python code. 
 
This example uses two tools for use within a LangChain agent: a Google Search tool 
and a Language Model tool acting specifically as a text summarizer. The Google Search 
tool, using the GoogleSearchAPIWrapper, will handle queries that involve finding recent 
event information. The Language Model tool leverages the capabilities of a language model 
to summarize texts. These tools are designed to be used interchangeably by the agent, 
depending on the nature of the user's query.

Notice how the agents used at first the “Search” tool to look for recent information 
about the Mars rover and then used the “Summarizer” tool for writing a summary.

LangChain provides an expansive toolkit that integrates various functions to improve 
the functionality of conversational agents. Here are some examples:

SerpAPI: This tool is an interface for the SerpAPI search engine, allowing the agent 
to perform robust online searches to pull in relevant data for a conversation or task.

PythonREPLTool: This unique tool enables the writing and execution of Python code within an agent. 
This opens up a wide range of possibilities for advanced computations and interactions within 
the conversation.

If you wish to add more specialized capabilities to your LangChain conversational agent, the platform 
offers the flexibility to create custom tools. By following the general tool creation guidelines 
provided in the LangChain documentation, you can develop tools tailored to the specific needs of your 
application.
"""

# We then instantiate an LLMChain specifically for text summarization
llm = OpenAI(model="text-davinci-003", temperature=0)

prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

# Next, we create the tools that our agent will use
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” OS env variables set 
# to be able to use Google Search via API
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    Tool(
       name='Summarizer',
       func=summarize_chain.run,
       description='useful for summarizing texts'
    )
]

# Create our agent that leverages two tools
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

# Run the agent with a question about summarizing the latest news about the Indian Lunar Lander
response = agent("What's the latest news about the Indian Lunar Lander? Then please summarize the results.")
print(response['output'])

import os
from dotenv import load_dotenv

load_dotenv()

"""
Here, the Agent is primarily using its internal knowledge to generate the output. Here's a brief 
explanation of how that works:

The agent receives a prompt to "Compose an epic science fiction saga about interstellar explorers.”
The agent then uses its understanding of language, narrative structure, and the specific themes mentioned in 
the prompt (science fiction, interstellar exploration, etc.) to generate a story.

LLM's understanding comes from its training data. It was trained on a diverse range of internet text, so it has 
a broad base of information to draw from. When asked to generate a science fiction story, it uses patterns it 
learned during training about how such stories are typically structured and what elements they usually contain.

Remember, even though the language model has vast training data to draw from, it doesn't "know" specific facts 
or have access to real-time information. Its responses are generated based on patterns learned during training, 
not from a specific knowledge database.
"""

from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["query"],
    template="You're a renowned science fiction writer. {query}"
)

# Initialize the language model
llm = OpenAI(model="text-davinci-003", temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)

tools = [
    Tool(
        name='Science Fiction Writer',
        func=llm_chain.run,
        description='Use this tool for generating science fiction stories. Input should be a command about generating specific types of stories.'
    )
]

# Initializing an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Testing the agent with the new prompt
response = agent.run("Compose an epic science fiction saga about interstellar explorers")
print(response)

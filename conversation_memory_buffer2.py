import os
from dotenv import load_dotenv

load_dotenv()

from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

"""
The ConversationChain uses the ConversationBufferMemory class by default to provide a 
history of messages. This memory can save the previous conversations in form of variables. 
The class accepts the return_messages argument which is helpful for dealing with chat models. 
This is how the CoversationChain keep context under the hood.
"""

memory = ConversationBufferMemory(return_messages=True)
memory.save_context(
    {"input": "hi there!"},
    {"output": "Hi there! It's nice to meet you. How can I help you today?"},
)

print(memory.load_memory_variables({}))

llm = OpenAI(model_name="text-davinci-003", temperature=0)
conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)


"""

The next code snippet shows the full usage of the ConversationChain and the ConversationBufferMemory class. 
Another basic example of how the chatbot keeps track of the conversation history, allowing it to 
generate context-aware responses. 

Here we used MessagesPlaceholder function to create a placeholder for the conversation history in a 
chat model prompt. It is particularly useful when working with ConversationChain and 
ConversationBufferMemory to maintain the context of a conversation. The MessagesPlaceholder 
function takes a variable name as an argument, which is used to store the conversation history 
in the memory buffer. We will cover that function later. 
"""
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "The following is a friendly conversation between a human and an AI."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

print(conversation.predict(input="Tell me a joke about elephants"))
print(conversation.predict(input="Who is the author of the Harry Potter series?"))
print(conversation.predict(input="What was the joke you told me earlier?"))


"""
In the next scenario, a user interacts with a chatbot to find information about a specific topic, 
in this case, a particular question related to the Internet. 
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "The following is a friendly conversation between a human and an AI."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)

"""
If we start with a general question:
"""
user_message = "Tell me about the history of the Internet."
response = conversation(user_message)
print(response)

"""
Here is the second query.
"""
# User sends another message
user_message = "Who are some important figures in its development?"
response = conversation(user_message)
print(
    response
)  # Chatbot responds with names of important figures, recalling the previous topic

"""
And the last query that showcase how using ConversationBufferMemory enables the chatbot to 
recall previous messages and provide more accurate and context-aware responses to the user's questions.
"""
user_message = "What did Tim Berners-Lee contribute?"
response = conversation(user_message)
print(response)

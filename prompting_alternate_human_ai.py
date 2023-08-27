import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

"""
In this strategy, few-shot prompting utilizes alternating human and AI messages. This technique can be 
especially beneficial for chat-oriented applications since the language model must comprehend the 
conversational context and provide appropriate responses.

While this approach effectively handles conversation context and is easy to implement for chat-based 
applications, it lacks flexibility for other application types and is limited to chat-based models. 
However, we can use alternating human/AI messages to create a chat prompt that translates English into 
pirate language.
"""

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template="You are a helpful assistant that translates english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run("I love programming.")
import os
from dotenv import load_dotenv

load_dotenv()

"""
This memory implementation stores the entire conversation history as a single string. The advantages of this 
approach is maintains a complete record of  the conversation, as well as being straightforward to implement and 
use. On the other hands, It can be less efficient as the conversation grows longer and may lead to excessive 
repetition if the conversation history is too long for the model's token limit.

If the token limit of the model is surpassed, the buffer gets truncated to fit within the model's token limit. 
This means that older interactions may be removed from the buffer to accommodate newer ones, and as a result, the 
conversation context might lose some information.

To avoid surpassing the token limit, you can monitor the token count in the buffer and manage the conversation 
accordingly. For example, you can choose to shorten the input texts or remove less relevant parts of the conversation 
to keep the token count within the model's limit.

First, as we learned in previous lesson, let’s observe how the ConversationBufferMemory can be used in the 
ConversationChain
"""

from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model_name="text-davinci-003", temperature=0)

conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)
conversation.predict(input="Hello!")

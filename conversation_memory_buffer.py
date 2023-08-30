import os
from dotenv import load_dotenv
load_dotenv()

"""
By default, LangChain's ConversationChain has a simple type of memory that remembers all 
previous inputs/outputs and adds them to the context that is passed. This can be considered a 
type of short-term memory. Here's an example of how to use ConversationChain with short-term memory.

As you can see from the “Current Conversation” section of the output, the model have access to all the previous 
messages. It can also remember what the initial message were after 3 questions.

The ConversationChain is a powerful tool that leverages past messages to produce fitting replies, resulting in 
comprehensive and knowledgeable outputs. This extra memory is invaluable when chatbots have to remember lots of 
details, especially when users ask for complicated information or engage in complex chats. By implementing the 
ConversationChain, users can enjoy seamless interactions with chatbots, ultimately enhancing their overall 
experience.
"""

from langchain import OpenAI, ConversationChain

llm = OpenAI(model_name="text-davinci-003", temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")

print(output)

output = conversation.predict(input="In what scenarios extra memory should be used?")
output = conversation.predict(input="There are various types of memory in Langchain. When to use which type?")
output = conversation.predict(input="Do you remember what was our first message?")

print(output)


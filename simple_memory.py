from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

"""
In LangChain, Memory refers to the mechanism that stores and manages the conversation history 
between a user and the AI. It helps maintain context and coherency throughout the interaction, 
enabling the AI to generate more relevant and accurate responses. Memory, such as 
ConversationBufferMemory, acts as a wrapper around ChatMessageHistory, extracting the messages 
and providing them to the chain for better context-aware generation.

In the output, you can see the memory being used by observing the "Current conversation" 
section. After each input from the user, the conversation is updated with both the user's input 
and the AI's response. This way, the memory maintains a record of the entire conversation. When 
the AI generates its next response, it will use this conversation history as context, making 
its responses more coherent and relevant.
"""
llm = OpenAI(model="text-davinci-003", temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# Continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

# Display the conversation
print(conversation)

import os
from dotenv import load_dotenv

load_dotenv()

"""
we will use the same logic and add the ConversationBufferMemory presented in the customer support 
chatbot using the same approach as in the previous example. This chatbot will handle basic inquiries 
about a fictional online store and maintain context throughout the conversation. The code below creates 
a prompt template for the customer support chatbot.
"""

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = OpenAI(model_name="text-davinci-003", temperature=0)

template = """You are a customer support chatbot for a highly advanced customer support AI 
for an online store called "Galactic Emporium," which specializes in selling unique,
otherworldly items sourced from across the universe. You are equipped with an extensive
knowledge of the store's inventory and possess a deep understanding of interstellar cultures. 
As you interact with customers, you help them with their inquiries about these extraordinary
products, while also sharing fascinating stories and facts about the cosmos they come from.

{chat_history}
Customer: {customer_input}
Support Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "customer_input"], template=template
)
chat_history = ""

convo_buffer = ConversationChain(llm=llm, memory=ConversationBufferMemory())

"""
The chatbot can handle customer inquiries and maintain context by storing the conversation 
history, allowing it to provide more coherent and relevant responses. You can access the 
prompt of any chain using the following naming convention.
"""
print(convo_buffer.prompt.template)

"""
Now, we will call the chatbot multiple times to imitate a user’s interaction that wants to get 
information about dog toys. We will only print the response of the final query. Still, you can 
read the history property and see how it saves all the previous queries (Human) and reponses (AI).
"""
convo_buffer("I'm interested in buying items from your store")
convo_buffer("I want toys for my pet, do you have those?")
convo_buffer("I'm interested in price of a chew toys, please")

"""
The cost of utilizing the AI model in ConversationBufferMemory is directly influenced by the number 
of tokens used in a conversation, thereby impacting the overall expenses. Large Language Models (LLMs) 
like ChatGPT have token limits, and the more tokens used, the more expensive the API requests become.

To calculate token count in a conversation, you can use the tiktoken package that counts the tokens 
for the messages passed to a model like gpt-3.5-turbo. Here's an example usage of the function for 
counting tokens in a conversation.
"""
import tiktoken


def count_tokens(text: str) -> int:
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = tokenizer.encode(text)
    return len(tokens)


conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020.",
    },
]

total_tokens = 0
for message in conversation:
    total_tokens += count_tokens(message["content"])

print(f"Total tokens in the conversation: {total_tokens}")


"""
For example, in a scenario where a conversation has a large sum of tokens, the computational 
cost and resources required for processing the conversation will be higher. This highlights 
the importance of managing tokens effectively. Strategies for achieving this include limiting 
memory size through methods like ConversationBufferWindowMemory or summarizing older interactions 
using ConversationSummaryBufferMemory. These approaches help control the token count while minimizing 
associated costs and computational demands in a more efficient manner.
"""

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

"""
The following example demonstrates how to create a chatbot using the GPT-4 model from OpenAI. 
After importing the necessary classes, we declare a set of messages. It starts by setting the 
context for the model (SystemMessage) that it is an assistant, followed by the user’s query 
(HumanMessage), and finishes by defining a sample response from the AI model (AIMessage)
"""
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
]

"""
When the user posed the question about the capital of France, 
the model confidently answered with "Paris.” Next up, we test 
if the model can leverage these discussions as a reference to 
delve further into details about the city without us explicitly 
mentioning the name (referring to Paris). The code below adds a 
new message which requires the model to understand and find the 
“city you just mentioned” reference from previous conversations.
"""
prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)

# add to messages
messages.append(prompt)

llm = ChatOpenAI(model_name="gpt-4")

response = llm(messages)

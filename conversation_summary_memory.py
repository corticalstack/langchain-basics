from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain import OpenAI

"""
ConversationSummaryBufferMemory is a memory management strategy that combines the ideas of keeping a buffer of 
recent interactions in memory and compiling old interactions into a summary. It extracts key information from 
previous interactions and condenses it into a shorter, more manageable format.  Here is a list of pros and 
cons of ConversationSummaryMemory.

Advantages:

Condensing conversation information
By summarizing the conversation, it helps reduce the number of tokens required to store the conversation history, 
which can be beneficial when working with token-limited models like GPT-3

Flexibility
You can configure this type of memory to return the history as a list of messages or as a plain text summary. This 
makes it suitable for chatbots.

Direct summary prediction
The predict_new_summary method allows you to directly obtain a summary prediction based on the list of messages and 
the previous summary. This enables you to have more control over the summarization process.


Disadvantages:

Loss of information
Summarizing the conversation might lead to a loss of information, especially if the summary is too short or omits 
important details from the conversation.

Increased complexity
Compared to simpler memory types like ConversationBufferMemory, which just stores the raw conversation history, 
ConversationSummaryMemoryrequires more processing to generate the summary, potentially affecting the performance 
of the chatbot. 

The summary memory is built on top of the ConversationChain. We use OpenAI's text-davinci-003 or other models like 
gpt-3.5-turbo to initialize the chain. This class uses a prompt template where the {history} parameter is feeding 
the information about the conversation history between the human and AI. 
"""
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# Create a ConversationChain with ConversationSummaryMemory
conversation_with_summary = ConversationChain(
    llm=llm, 
    memory=ConversationSummaryMemory(llm=llm),
    verbose=True
)

# Example conversation
response = conversation_with_summary.predict(input="Hi, what's up?")
print(response)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    The following is a friendly conversation between a human and an AI. The AI is 
    talkative and provides lots of specific details from its context. If the AI does 
    not know the answer to a question, it truthfully says it does not know.
    \nCurrent conversation:\n{topic}
    """,
)

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

llm = OpenAI(temperature=0)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40),
    verbose=True
)
conversation_with_summary.predict(input="Hi, what's up?")
conversation_with_summary.predict(input="Just working on writing some documentation!")
response = conversation_with_summary.predict(input="For LangChain! Have you heard of it?")
print(response)



"""
Recap and Strategies
If the ConversationBufferMemory surpasses the token limit of the model, you will receive an error, as the 
model will not be able to handle the conversation with the exceeded token count.

To manage this situation, you can adopt different strategies:

Remove oldest messages

 One approach is to remove the oldest messages in the conversation transcript once the token count is reached. 
 This method can cause the conversation quality to degrade over time, as the model will gradually lose the context 
 of the earlier portions of the conversation.

Limit conversation duration
Another approach is to limit the conversation duration to the max token length or a certain number of turns. Once 
the max token limit is reached and the model would lose context if you were to allow the conversation to continue, 
you can prompt the user that they need to begin a new conversation and clear the messages array to start a brand 
new conversation with the full token limit available.

ConversationBufferWindowMemory Method:
This method limits the number of tokens being used by maintaining a fixed-size buffer window that stores only the 
most recent tokens, up to a specified limit. 

→This is suitable for remembering recent interactions but not distant ones.

ConversationSummaryBufferMemory Approach:

This method combines the features: of ConversationSummaryMemoryand ConversationBufferWindowMemory.
It summarizes the earliest interactions in a conversation while maintaining the most recent tokens in their raw, 
information-rich form, up to a specified limit.

→This allows the model to remember both distant and recent interactions but may require more tweaking on what to 
summarize and what to maintain within the buffer window.

It's important to keep track of the token count and only send the model a prompt that falls within the token limit. 

→You can use OpenAI's tiktoken library to handle the token count efficiently

Token limit:

The maximum token limit for the GPT-3.5-turbo model is 4096 tokens. This limit applies to both the input and output 
tokens combined. If the conversation has too many tokens to fit within this limit, you will have to truncate, omit, 
or shrink the text until it fits. Note that if a message is removed from the message's input, the model will lose 
all knowledge of it. 

→To handle this situation, you can split the input text into smaller chunks and process them separately or adopt 
other strategies to truncate, omit, or shrink the text until it fits within the limit. One way to work with large 
texts is to use batch processing. This technique involves breaking down the text into smaller chunks and processing 
each batch separately while providing some context before and after the text to edit. You can find out more about 
this technique here: 

See following blog:
https://marco-gonzalez.medium.com/breaking-the-token-limit-how-to-work-with-large-amounts-of-text-in-chatgpt-da18c798d882

When choosing a conversational memory implementation for your LangChain chatbot, consider factors such as 
conversation length, model token limits, and the importance of maintaining the full conversation history. Each 
type of memory implementation offers unique benefits and trade-offs, so it's essential to select the one that 
best suits your chatbot's requirements.
"""
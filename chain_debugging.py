"""
It is possible to trace the inner workings of any chain by setting the verbose argument to True. 
As you can see in the following code, the chain will return the initial prompt and the output. 
The output depends on the application. It may contain more information if there are more steps.
"""
from langchain import ConversationChain, PromptTemplate


template = """List all possible words as substitute for 'artificial' as comma separated.

Current conversation:
{history}

{input}"""

conversation = ConversationChain(
    llm=llm,
    prompt=PromptTemplate(
        template=template,
        input_variables=["history", "input"],
        output_parser=output_parser,
    ),
    memory=ConversationBufferMemory(),
    verbose=True,
)

conversation.predict(input="")

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

load_dotenv()

"""
The LLMChain works as follows:

1. Takes (multiple) input variables.
2. Uses the PromptTemplate to format the input variables into a prompt.
3. Passes the formatted prompt to the model (LLM or ChatModel).
4. If an output parser is provided, it uses the OutputParser to parse the output of the LLM into a final format.
"""
llm = OpenAI(model="text-davinci-003", temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("eco-friendly water bottles"))

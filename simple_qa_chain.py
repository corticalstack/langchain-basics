from dotenv import load_dotenv

load_dotenv()

"""
We define a custom prompt template by creating an instance of the PromptTemplate class. 
The template string contains a placeholder {question} for the input question, followed by 
a newline character and the "Answer:" label.  The input_variables argument is set to the 
list of available placeholders in the prompt (like a question in this case) to indicate 
the name of the variable that the chain will replace in the template.run() method.

We then instantiate an OpenAI model named text-davinci-003 with a temperature of 0. 
The OpenAI class is used to create the instance, and the model_name and temperature 
arguments are provided. Finally, we create a question-answering chain using the LLMChain 
class. 

The class constructor takes two arguments: llm, which is the instantiated OpenAI model, 
and prompt, which is the custom prompt template we defined earlier. 
"""
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

prompt = PromptTemplate(
    template="Question: {question}\nAnswer:", input_variables=["question"]
)

llm = OpenAI(model_name="text-davinci-003", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("what is the meaning of life?"))

from dotenv import load_dotenv

load_dotenv()

from langchain import PromptTemplate, OpenAI, LLMChain

prompt_template = "What is a word to replace the following: {word}?"

# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = OpenAI(model_name="text-davinci-003", temperature=0)


"""
Several methods are available for utilizing a chain, each yielding a distinct output format. The example in 
this section is creating a bot that can suggest a replacement word based on context. The code snippet below 
demonstrates the utilization of the GPT-3 model through the OpenAI API. It generates a prompt using the 
PromptTemplate from LangChain, and finally, the LLMChain class ties all the components. 
"""
llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))


"""
The most straightforward approach uses the chain class __call__ method. It means passing the input directly 
to the object while initializing it. It will return the input variable and the model’s response under the 
text key.
"""
print(llm_chain("artificial"))

"""
It is also possible to use the .apply() method to pass multiple inputs at once and receive a list for 
each input. The sole difference lies in the exclusion of inputs within the returned list. Nonetheless, 
the returned list will maintain the identical order as the input.
"""
input_list = [{"word": "artificial"}, {"word": "intelligence"}, {"word": "robot"}]

print(llm_chain.apply(input_list))

"""
The .generate() method will return an instance of LLMResult, which provides more information. For example, 
the finish_reason key indicates the reason behind the stop of the generation process. It could be stopped, 
meaning the model decided to finish or reach the length limit. There is other self-explanatory information 
like the number of total used tokens or the used model.
"""
print(llm_chain.generate(input_list))


"""
The next method we will discuss is .predict(). (which could be used interchangeably with .run()) Its best 
use case is to pass multiple inputs for a single prompt. However, it is possible to use it with one input 
variable as well. The following prompt will pass both the word we want a substitute for and the context 
the model must consider.
"""
prompt_template = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template=prompt_template, input_variables=["word", "context"]
    ),
)

print(llm_chain.predict(word="fan", context="object"))
# or llm_chain.run(word="fan", context="object")

"""
The model correctly suggested that a Ventilator would be a suitable replacement for the word fan in the context of 
objects. Furthermore, when we repeat the experiment with a different context, humans, the output will change the Admirer.
"""
print(llm_chain.predict(word="fan", context="humans"))
# or llm_chain.run(word="fan", context="humans")

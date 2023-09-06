import os
from dotenv import load_dotenv

from langchain import LLMChain, FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", temperature=0)

"""
The template is a formatted string with a {query} placeholder that will be substituted with a real question when applied. 
To create a PromptTemplate object, two arguments are required:

input_variables: A list of variable names in the template; in this case, it includes only the query.
template: The template string containing formatted text and placeholders.

After creating the PromptTemplate object, it can be used to produce prompts with specific questions by providing input data. 
The input data is a dictionary where the key corresponds to the variable name in the template. The resulting prompt can then 
be passed to a language model to generate answers.

For more advanced usage, you can create a FewShotPromptTemplate with an ExampleSelector to select a subset of examples that 
will be most informative for the language model.
"""

examples = [
    {"animal": "lion", "habitat": "savanna"},
    {"animal": "polar bear", "habitat": "Arctic ice"},
    {"animal": "elephant", "habitat": "African grasslands"},
]

example_template = """
Animal: {animal}
Habitat: {habitat}
"""

example_prompt = PromptTemplate(
    input_variables=["animal", "habitat"], template=example_template
)

dynamic_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Identify the habitat of the given animal",
    suffix="Animal: {input}\nHabitat:",
    input_variables=["input"],
    example_separator="\n\n",
)

# Create the LLMChain for the dynamic_prompt
chain = LLMChain(llm=llm, prompt=dynamic_prompt)

# Run the LLMChain with input_data
input_data = {"input": "tiger"}
response = chain.run(input_data)

print(response)

# You can also save your PromptTemplate to a file in your local filesystem in JSON or YAML format:
example_prompt.save("awesome_prompt.json")

# And load it back from the file:
from langchain.prompts import load_prompt

loaded_prompt = load_prompt("awesome_prompt.json")

dynamic_prompt.save("awesome_dynamic_prompt.json")
loaded_dynamic_prompt = load_prompt("awesome_dynamic_prompt.json")

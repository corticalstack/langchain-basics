import os
from dotenv import load_dotenv
load_dotenv()

"""
Manages comma-separated outputs. It handles one specific case: anytime you want to receive a list of outputs from the model.
The parser does not require a setting up step. Therefore it is less flexible. We can create the object by calling the class.

Although most of the sample code has been explained in the previous subsection, two parts might need attention. Firstly, we 
tried a new format for the prompt’s template to show different ways to write a prompt. Secondly, the use of .format() instead 
of .format_prompt() to generate the model’s input. The main difference compared to the previous subsection’s code is that we 
no longer need to call the .to_string() object since the prompt is already in string type.

As you can see, the final output is a list of words that has some overlaps with the PydanticOutputParser approach with more variety. 
However, requesting additional reasoning information using the CommaSeparatedOutputParser class is impossible.

"""

from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Prepare the Prompt
template = """
Offer a list of suggestions to substitute the word '{target_word}' based the presented the following text: {context}.
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format(
  target_word="behaviour",
  context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# Loading OpenAI API
model = OpenAI(model_name='text-davinci-003', temperature=0.0)

# Send the Request
output = model(model_input)
print(parser.parse(output))

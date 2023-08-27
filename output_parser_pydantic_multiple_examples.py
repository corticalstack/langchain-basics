import os
from dotenv import load_dotenv
load_dotenv()

"""
Here is a sample code for Pydantic class to process multiple outputs. It requests the model to 
suggest a list of words and present the reasoning behind each proposition.
Asks the model to present its reasoning, and the suggestion class declares a new output named reasons. 
Also, the validator function manipulates the output to ensure every reasoning ends with a dot. Another 
use case of the validator function could be output manipulation.
"""

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, FieldValidationInfo, field_validator
from typing import List

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")
    
    @field_validator('words')
    def not_start_with_number(cls, info: FieldValidationInfo):
      for item in info:
        if item[0].isnumeric():
          raise ValueError("The word can not start with numbers!")
      return info
    
    @field_validator('reasons')
    def end_with_dot(cls, info: FieldValidationInfo):
      for idx, item in enumerate(info):
        if item[-1] != ".":
          info[idx] += "."
      return info

parser = PydanticOutputParser(pydantic_object=Suggestions)

from langchain.prompts import PromptTemplate

template = """
Offer a list of suggestions to substitute the specified target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
			target_word="behaviour",
			context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

from langchain.llms import OpenAI

model = OpenAI(model_name='text-davinci-003', temperature=0.0)

output = model(model_input.to_string())

print(parser.parse(output))



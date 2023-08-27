import os
from dotenv import load_dotenv
load_dotenv()

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, FieldValidationInfo, field_validator
from typing import List

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitute words based on context")

    # Throw error in case of receiving a numbered-list from API
    @field_validator('words')
    def not_start_with_number(cls, info: FieldValidationInfo):
        for item in info:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return info

parser = PydanticOutputParser(pydantic_object=Suggestions)

from langchain.prompts import PromptTemplate

template = """
Offer list of suggestions to substitute the specified target_word based the presented context.
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



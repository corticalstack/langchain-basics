import os
from dotenv import load_dotenv

load_dotenv()

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


# Define data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitute words based on context")
    reasons: List[str] = Field(
        description="the reasoning of why this word fits the context"
    )


parser = PydanticOutputParser(pydantic_object=Suggestions)

# Define prompt
template = """
Offer a list of suggestions to substitute the specified target_word based the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model_input = prompt.format_prompt(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson.",
)

# Define Model
model = OpenAI(model_name="text-davinci-003", temperature=0.0)

from langchain.output_parsers import RetryWithErrorOutputParser

missformatted_output = '{"words": ["conduct", "manner"]}'

retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)

print(retry_parser.parse_with_prompt(missformatted_output, model_input))

import os
from dotenv import load_dotenv

load_dotenv()

"""
This method tries to fix the parsing error by looking at the model’s response and 
the previous parser. It uses a Large Language Model (LLM) to solve the issue. We will 
use GPT-3 to be consistent with the rest of the lesson, but it is possible to pass any 
supported model. Let’s start by defining the Pydantic data schema and show a sample 
error that could occur.

Note!: The following approaches work with the PydanticOutputParser class since it is 
the only one with a validation method.
"""

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(
        description="the reasoning of why this word fits the context"
    )


try:
    parser = PydanticOutputParser(pydantic_object=Suggestions)

    missformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'

    print(parser.parse(missformatted_output))
except Exception as e:
    pass


"""
As you can see in the error message, the parser correctly identified an error in our sample response 
(missformatted_output) since we used the word reasoning instead of the expected reasons key. The 
OutputFixingParser class could easily fix this error.
"""
from langchain.llms import OpenAI
from langchain.output_parsers import OutputFixingParser

model = OpenAI(model_name="text-davinci-003", temperature=0.0)

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
print(outputfixing_parser.parse(missformatted_output))

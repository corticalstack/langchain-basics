import os
from dotenv import load_dotenv
load_dotenv()

"""
This is the first output parser implemented by the LangChain team. 
While it can process multiple outputs, it only supports texts and does not 
provide options for other data types, such as lists or integers. It can be 
used when you want to receive one response from the model. For example, 
only one substitute word in the thesaurus application.
"""
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="words", description="A substitute word based on context"),
    ResponseSchema(name="reasons", description="the reasoning of why this word fits the context.")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

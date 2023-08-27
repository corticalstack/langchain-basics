import os
from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

"""
We created an instance of the RecursiveCharacterTextSplitter class with the desired parameters. 
The default list of characters to split by is ["\n\n", "\n", " ", ""].

The text is first split by two new-line characters (\n\n). Then, since the chunks are still larger 
than the desired chunk size (50), the class tries to split the output by a single new-line character (\n).

In this example, the text is loaded from a file, and the RecursiveCharacterTextSplitter is used to 
split it into chunks with a maximum size of 50 characters and an overlap of 10 characters. The output 
will be a list of documents containing the split text.

To use a token counter, you can create a custom function that calculates the number of tokens in a 
given text and pass it as the length_function parameter. This will ensure that your text splitter calculates 
the length of chunks based on the number of tokens instead of the number of characters.
"""

loader = PyPDFLoader("The One Page Linux Manual.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
)

docs = text_splitter.split_documents(pages)
for doc in docs:
    print(doc)
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import SpacyTextSplitter

"""
The SpacyTextSplitter helps split large text documents into smaller chunks based on a specified size. 
This is useful for better management of large text inputs. It's important to note that the SpacyTextSplitter 
is an alternative to NLTK-based sentence splitting. You can create a SpacyTextSplitter object by specifying 
the chunk_size parameter, measured by a length function passed to it, which defaults to the number of characters.
"""

# Load a long document
with open('/home/cloudsuperadmin/scrape-chain/langchain/LLM.txt', encoding= 'unicode_escape') as f:
    sample_text = f.read()

# Instantiate the SpacyTextSplitter with the desired chunk size
text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=20)

# Split the text using SpacyTextSplitter
texts = text_splitter.split_text(sample_text)

# Print the first chunk
print(texts[0])


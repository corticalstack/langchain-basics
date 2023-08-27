import os
from dotenv import load_dotenv
load_dotenv()

"""
The main advantage of using TokenTextSplitter over other text splitters, like 
CharacterTextSplitter, is that it respects the token boundaries, ensuring that 
the chunks do not split tokens in the middle. This can be particularly helpful 
in maintaining the semantic integrity of the text when working with language 
models and embeddings.

This type of splitter breaks down raw text strings into smaller pieces by initially 
converting the text into BPE (Byte Pair Encoding) tokens, and subsequently dividing 
these tokens into chunks. It then reassembles the tokens within each chunk back into 
text. The tiktoken python package is required for using this class. (pip install -q tiktoken)


The chunk_size parameter sets the maximum number of BPE tokens in each chunk, while chunk_overlap 
defines the number of overlapping tokens between adjacent chunks. By modifying these parameters, 
you can fine-tune the granularity of the text chunks.

One potential drawback of using TokenTextSplitter is that it may require additional computation 
when converting text to BPE tokens and back. If you need a faster and simpler text-splitting method, 
you might consider using CharacterTextSplitter, which directly splits the text based on character 
count, offering a more straightforward approach to text segmentation.
"""

from langchain.text_splitter import TokenTextSplitter

# Load a long document
with open('/home/cloudsuperadmin/scrape-chain/langchain/LLM.txt', encoding= 'unicode_escape') as f:
    sample_text = f.read()

# Initialize the TokenTextSplitter with desired chunk size and overlap
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=50)

# Split into smaller chunks
texts = text_splitter.split_text(sample_text)
print(texts[0])


import os
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import NLTKTextSplitter

# Load a long document
with open('/home/cloudsuperadmin/scrape-chain/langchain/LLM.txt', encoding= 'unicode_escape') as f:
    sample_text = f.read()

text_splitter = NLTKTextSplitter(chunk_size=500)
texts = text_splitter.split_text(sample_text)
print(texts)


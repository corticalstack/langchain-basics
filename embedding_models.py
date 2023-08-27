from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings

"""
We initialize our embedding model. For this task, we've chosen the pre-trained 
"sentence-transformers/all-mpnet-base-v2" model. This model is designed to transform 
sentences into embeddings - vectors that encapsulate the semantic meaning of the sentences. 
The model_kwargs parameter is used here to specify that we want our computations to be performed 
on the CPU.

Before executing the subsequent code, make sure to install the Sentence Transformer library 
by using the command pip install sentence_transformers===2.2.2. This library offers powerful 
pre-trained models designed to generate embedding representations.
"""

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

documents = ["Document 1", "Document 2", "Document 3"]
doc_embeddings = hf.embed_documents(documents)
print(doc_embeddings)

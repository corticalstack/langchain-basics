from dotenv import load_dotenv

load_dotenv()

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

"""
Uses LangChains GoogleSearchAPIWrapper class to receive search results. It works in combination 
with the Tool class that presents the utilities for agents to help them interact with the outside 
world. In this case, creating a tool out of any function, like top_n_results is possible. The API 
will return the pages title, URL, and a short description.
"""

search = GoogleSearchAPIWrapper()
TOP_N_RESULTS = 10


def top_n_results(query):
    return search.results(query, TOP_N_RESULTS)


tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=top_n_results,
)

query = "What is the latest fast and furious movie?"

results = tool.run(query)

for result in results:
    print(result["title"])
    print(result["link"])
    print(result["snippet"])
    print("-" * 50)

"""
Now, we use the results variable’s link key to download and parse the contents. The newspaper 
library takes care of everything. However, it might be unable to capture some contents in certain 
situations, like anti-bot mechanisms or having a file as a result.
"""
import newspaper

pages_content = []

for result in results:
    try:
        article = newspaper.Article(result["link"])
        article.download()
        article.parse()
        if len(article.text) > 0:
            pages_content.append({"url": result["link"], "text": article.text})
    except:
        continue

"""
Process the Search Results
We now have the top 10 results from the Google search. (Honestly, who looks at Google’s second page?) 
However, it is not efficient to pass all the contents to the model because of the following reasons:

The model’s context length is limited.
It will significantly increase the cost if we process all the search results.
In almost all cases, they share similar pieces of information.
So, let’s find the most relevant results, 

Incorporating the LLMs embedding generation capability will enable us to find contextually similar content. 
It means converting the text to a high-dimensionality tensor that captures meaning. The cosine similarity 
function can find the closest article with respect to the user’s question.

It starts by splitting the texts using the RecursiveCharacterTextSplitter class to ensure the content 
lengths are inside the model’s input length. The Document class will create a data structure from each 
chunk that enables saving metadata like URL as the source. The model can later use this data to know the 
content’s location.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)

docs = []
for d in pages_content:
    chunks = text_splitter.split_text(d["text"])
    for chunk in chunks:
        new_doc = Document(page_content=chunk, metadata={"source": d["url"]})
        docs.append(new_doc)

"""
The subsequent step involves utilizing the OpenAI API's OpenAIEmbeddings class, specifically 
the .embed_documents() method for search results and the .embed_query() method for the user's 
question, to generate embeddings.
"""
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

docs_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])
query_embedding = embeddings.embed_query(query)

"""
Lastly, the get_top_k_indices function accepts the content and query embedding vectors and 
returns the index of top K candidates with the highest cosine similarities to the user's 
request. Later, we use the indexes to retrieve the best-fit documents
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_top_k_indices(list_of_doc_vectors, query_vector, top_k):
    # convert the lists of vectors to numpy arrays
    list_of_doc_vectors = np.array(list_of_doc_vectors)
    query_vector = np.array(query_vector)

    # compute cosine similarities
    similarities = cosine_similarity(
        query_vector.reshape(1, -1), list_of_doc_vectors
    ).flatten()

    # sort the vectors based on cosine similarity
    sorted_indices = np.argsort(similarities)[::-1]

    # retrieve the top K indices from the sorted list
    top_k_indices = sorted_indices[:top_k]

    return top_k_indices


top_k = 2
best_indexes = get_top_k_indices(docs_embeddings, query_embedding, top_k)
best_k_documents = [doc for i, doc in enumerate(docs) if i in best_indexes]

"""
Chain with Source: Finally, we used the selected articles in our prompt (using the 
stuff method) to assist the model in finding the correct answer. LangChain provides 
the load_qa_with_sources_chain() chain, which is designed to accept a list of input_documents 
as a source of information and a question argument which is the user’s question. The final 
part involves preprocessing the model’s response to extract its answer and the sources it utilized.
"""

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="stuff")

response = chain(
    {"input_documents": best_k_documents, "question": query}, return_only_outputs=True
)

response_text, response_sources = response["output_text"].split("SOURCES:")
response_text = response_text.strip()
response_sources = response_sources.strip()

print(f"Answer: {response_text}")
print(f"Sources: {response_sources}")

"""
The use of search results helped the model find the correct answer, even though it never 
saw it before during the training stage. The question and answering chain with source also 
provides information regarding the sources utilized by the model to derive the answer.
"""

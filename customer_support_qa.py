import os
from dotenv import load_dotenv

load_dotenv()

"""
As we witness accelerated technological progress, large language models like GPT-4 and ChatGPT have emerged 
as significant breakthroughs in the tech landscape. These state-of-the-art models demonstrate exceptional 
prowess in content generation. However, they are not without their share of challenges, such as biases and 
hallucinations. Despite these limitations, LLMs have the potential to bring about a transformative impact on chatbot development.

Traditional, primarily intent-based chatbots have been designed to respond to specific user intents. These intents 
comprise a collection of sample questions and corresponding responses. For instance, a "Restaurant Recommendations" 
intent might include sample questions like "Can you suggest a good Italian restaurant nearby?" or "Where can I find the 
best sushi in town?" with responses such as "You can try the Italian restaurant 'La Trattoria' nearby" or "The top-rated 
sushi place in town is 'Sushi Palace.'"

When users interact with the chatbot, their queries are matched to the most similar intent, generating the associated response. 
However, as LLMs continue to evolve, chatbot development is shifting toward more sophisticated and dynamic solutions capable of 
handling a broader range of user inquiries with greater precision and nuance.
"""

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate

# we'll use information from the following articles
urls = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-spotify-account/",
    "https://beebom.com/how-download-gif-twitter/",
    "https://beebom.com/how-use-chatgpt-linux-terminal/",
    "https://beebom.com/how-delete-spotify-account/",
    "https://beebom.com/how-save-instagram-story-with-music/",
    "https://beebom.com/how-install-pip-windows/",
    "https://beebom.com/how-check-disk-usage-linux/",
]

# We load the documents from the provided URLs and split them into chunks using the CharacterTextSplitter
# with a chunk size of 1000 and no overlap:

# use the selenium scraper to load the documents
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()

# we split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)


"""
Next, we compute the embeddings using OpenAIEmbeddings and store them in a Deep Lake vector store on the 
cloud. In an ideal production scenario, we could upload a whole website or course lesson on a Deep Lake 
dataset, allowing for search among even thousands or millions of documents. As we are using a cloud 
serverless Deep Lake dataset, applications running on different locations can easily access the same 
centralized dataset without the need of deploying a vector store on a custom machine.
"""
# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# create Deep Lake dataset
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)

# let's see the top relevant documents to a specific query
query = "how to check disk usage in linux?"
docs = db.similarity_search(query)
print(docs[0].page_content)


"""
We will create a prompt template that incorporates role-prompting, relevant Knowledge Base information, and the user's question:
let's write a prompt for a customer support chatbot that answer questions using information extracted from our db

The template sets the chatbot's persona as an exceptional customer support chatbot. The template takes 
two input variables: chunks_formatted, which consists of the pre-formatted chunks from articles, and query, 
representing the customer's question. The objective is to generate an accurate answer using only the provided 
chunks without creating any false or invented information.
"""

template = """You are an exceptional customer support chatbot that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chunks_formatted", "query"],
    template=template,
)

"""
To generate a response, we first retrieve the top-k (e.g., top-3) chunks most similar to the 
user query, format the prompt, and send the formatted prompt to the GPT3 model with a temperature of 0.
"""
# the full pipeline

# user question
query = "How to check disk usage in linux?"

# retrieve relevant chunks
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

# format the prompt
chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

# generate answer
llm = OpenAI(model="text-davinci-003", temperature=0)
answer = llm(prompt_formatted)
print(answer)

"""
Issues with Generating Answers using GPT-3
In the previous example, the chatbot generally performs well. However, there are certain situations where it could fail.

Suppose we ask, "Is the Linux distribution free?" and provide GPT-3 with a document about kernel features as context. 
It might generate an answer like "Yes, the Linux distribution is free to download and use," even if such information 
is not present in the context document. Producing false information is highly undesirable for customer service chatbots!

GPT-3 is less likely to generate false information when the answer to the user's question is contained within the 
context. Since user questions are often brief and ambiguous, we cannot always rely on the semantic search step to 
retrieve the correct document. Thus, there is always a risk of generating false information.

GPT-3 is highly effective in creating conversational chatbots capable of answering specific questions based on the 
contextual information provided in the prompt. However, it can be challenging to ensure that the model generates answers 
solely based on the context, as it has a tendency to hallucinate (i.e., generate new, potentially false information). The 
severity of generating false information varies depending on the use case.

To conclude, we implemented a context-aware question-answering system using LangChain, following the provided code 
and strategies. The process involved splitting documents into chunks, computing their embeddings, implementing a retriever 
to find similar chunks, crafting a prompt for GPT-3, and using the GPT3 model for text generation. This approach demonstrates 
the potential of leveraging GPT-3 to create powerful and contextually accurate chatbots while also highlighting the need to 
be cautious about the possibility of generating false information.
"""

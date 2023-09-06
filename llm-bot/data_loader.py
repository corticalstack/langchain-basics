import os
from dotenv import load_dotenv

load_dotenv()

from langchain.utilities import ApifyWrapper
from langchain.document_loaders.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import DeepLake


# Prep the knowledgebase
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_knowledgebase"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

try:
    DeepLake.force_delete_by_path(dataset_path)
except:
    pass

embeddings = CohereEmbeddings(model="embed-english-v2.0")

dbs = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# Scrape the website and load the data
apify = ApifyWrapper()
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://docs.langchain.com/docs/"}]},
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"]
        if dataset_item["text"]
        else "No content available",
        metadata={
            "source": dataset_item["url"],
            "title": dataset_item["metadata"]["title"],
        },
    ),
)

docs = loader.load()

# we split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20, length_function=len
)
docs_split = text_splitter.split_documents(docs)

dbs.add_documents(docs_split)

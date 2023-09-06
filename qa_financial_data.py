import os
from dotenv import load_dotenv

load_dotenv()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PagedPDFSplitter

import requests
import tqdm
from typing import List

# financial reports of amamzon, but can be replaced by any URLs of pdfs
urls = [
    "https://s2.q4cdn.com/299287126/files/doc_financials/2020/Q1/AMZN-Q1-2020-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2020/q2/Q2-2020-Amazon-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2020/q4/Amazon-Q4-2020-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2021/q1/Amazon-Q1-2021-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2021/q2/AMZN-Q2-2021-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2021/q3/Q3-2021-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2021/q4/business_and_financial_update.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2022/q1/Q1-2022-Amazon-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2022/q2/Q2-2022-Amazon-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2022/q3/Q3-2022-Amazon-Earnings-Release.pdf",
    "https://s2.q4cdn.com/299287126/files/doc_financials/2022/q4/Q4-2022-Amazon-Earnings-Release.pdf",
]


def load_reports(urls: List[str]) -> List[str]:
    """Load pages from a list of urls"""
    pages = []

    for url in tqdm.tqdm(urls):
        r = requests.get(url)
        path = url.split("/")[-1]
        with open(path, "wb") as f:
            f.write(r.content)
        loader = PagedPDFSplitter(path)
        local_pages = loader.load_and_split()
        pages.extend(local_pages)
    return pages


pages = load_reports(urls)

# We now use the Text Splitter Util to split documents into pages.
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings()

my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_chat_with_finance"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
db.add_documents(texts)

optional_params = {
    "top_p": 0.95,
}

llm = ChatOpenAI(
    model_name="gpt-4",
    model_kwargs=optional_params,
    temperature=0,
    streaming=True,
    verbose=True,
    max_tokens=1500,
)
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

print(qa.run("Combine total revenue in 2020?"))

print(qa.run("What is the revenue in 2021 Q3?"))

print(qa.run("What is the revenue in 2023 Q1?"))

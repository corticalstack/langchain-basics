import json
from dotenv import load_dotenv

load_dotenv()

import requests
from newspaper import Article

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
}

article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

session = requests.Session()

try:
    response = session.get(article_url, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        print(f"Title: {article.title}")
        print(f"Text: {article.text}")

    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")


"""
Imports essential classes and functions from the LangChain and sets up a 
ChatOpenAI instance with a temperature of 0 for controlled response generation. 
Additionally, it imports chat-related message schema classes, which enable the 
smooth handling of chat-based tasks. The following code will start by setting 
the prompt and filling it with the article’s content.
"""

from langchain.schema import HumanMessage

# we get the article data from the scraping part
article_title = article.title
article_text = article.text

# prepare template for prompt
template = """You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

Write a summary of the previous article.
"""

prompt = template.format(article_title=article.title, article_text=article.text)

messages = [HumanMessage(content=prompt)]

"""
The HumanMessage is a structured data format representing user messages within the 
chat-based interaction framework. The ChatOpenAI class is utilized to interact with 
the AI model, while the HumanMessage schema provides a standardized representation 
of user messages. The template consists of placeholders for the article's title and
content, which will be substituted with the actual article_title and article_text. 
This process simplifies and streamlines the creation of dynamic prompts by allowing 
you to define a template with placeholders and then replace them with actual data 
when needed.
"""

from langchain.chat_models import ChatOpenAI

# load the model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

"""
As we loaded the model and set the temperature to 0. We’d use the chat() instance 
to generate a summary by passing a single HumanMessage object containing the formatted 
prompt. The AI model processes this prompt and returns a concise summary:
"""

# generate summary
summary = chat(messages)
print(summary.content)

"""
If we want a bulleted list, we can modify a prompt and get the result.
"""
# prepare template for prompt
template = """You are an advanced AI assistant that summarizes online articles into bulleted lists.

Here's the article you need to summarize.

==================
Title: {article_title}

{article_text}
==================

Now, provide a summarized version of the article in a bulleted list format.
"""

# format prompt
prompt = template.format(article_title=article.title, article_text=article.text)

# generate summary
summary = chat([HumanMessage(content=prompt)])
print(summary.content)


"""
If you want to get the summary in French, you can instruct the model to generate 
the summary in French language. However, please note that GPT-4's main training language 
is English and while it has a multilingual capability, the quality may vary for languages
other than English. Here's how you can modify the prompt.
"""

# prepare template for prompt
template = """You are an advanced AI assistant that summarizes online articles into bulleted lists in French.

Here's the article you need to summarize.

==================
Title: {article_title}

{article_text}
==================

Now, provide a summarized version of the article in a bulleted list format, in French.
"""

# format prompt
prompt = template.format(article_title=article.title, article_text=article.text)

# generate summary
summary = chat([HumanMessage(content=prompt)])
print(summary.content)

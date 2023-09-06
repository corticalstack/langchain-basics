import os
from dotenv import load_dotenv

load_dotenv()

"""
The MarkdownTextSplitter is designed to split text written using Markdown languages like headers, 
code blocks, or dividers. It is implemented as a simple subclass of RecursiveCharacterSplitter 
with Markdown-specific separators. By default, these separators are determined by the Markdown 
syntax, but they can be customized by providing a list of characters during the initialization 
of the MarkdownTextSplitter instance. The chunk size, which is initially set to the number of 
characters, is measured by the length function passed in. To customize the chunk size, provide 
an integer value when initializing an instance.
"""
from langchain.text_splitter import MarkdownTextSplitter

markdown_text = """
# 

# Welcome to My Blog!

## Introduction
Hello everyone! My name is **John Doe** and I am a _software developer_. I specialize in Python, Java, and JavaScript.

Here's a list of my favorite programming languages:

1. Python
2. JavaScript
3. Java

You can check out some of my projects on [GitHub](https://github.com).

## About this Blog
In this blog, I will share my journey as a software developer. I'll post tutorials, my thoughts on the latest technology trends, and occasional book reviews.

Here's a small piece of Python code to say hello:

\``` python
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("John")
\```

Stay tuned for more updates!

## Contact Me
Feel free to reach out to me on [Twitter](https://twitter.com) or send me an email at johndoe@email.com.

"""

markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
docs = markdown_splitter.create_documents([markdown_text])

print(docs)

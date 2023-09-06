import os
from dotenv import load_dotenv

load_dotenv()

"""
We set up the environment and retrieved the news article.

Install required libraries: The first step is to ensure that the necessary libraries, namely requests, newspaper3k, and LangChain, are installed.
Scrape articles: We will use the requests library to scrape the content of the target news articles from their respective URLs.
Extract titles and text: The newspaper library will be used to parse the scraped HTML, extracting the titles and text of the articles.
Preprocess the text: The extracted texts need to be cleaned and preprocessed to make them suitable for input to LLM.
The rest of the lesson will explore new possibilities to enhance the application’s performance further.

Use Few-Shot Learning Technique: We use the few-shot learning technique in this step. This template will provide a few examples of the language model to guide it in generating the summaries in the desired format - a bulleted list.
Generate summaries: With the modified prompt, we utilize the model to generate concise summaries of the extracted articles' text in the desired format.
Use the Output Parsers: We employ the Output Parsers to interpret the output from the language model, ensuring it aligns with our desired structure and format.
Output the results: Finally, we present the bulleted summaries along with the original titles, enabling users to quickly grasp the main points of each article in a structured manner.
"""

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

from langchain.schema import HumanMessage

# we get the article data from the scraping part
article_title = article.title
article_text = article.text

"""
Provide the model with a better understanding of how we want it to respond.  Here we have a few important components:

Article data: The title and text of the article are obtained, which will be used as inputs to the model.

Template preparation: A template is prepared for the prompt. This template includes a few-shot learning style, 
where the model is provided with examples of how it has previously converted articles into a bulleted list format. 
The template also includes placeholders for the actual article title and text that will be summarized.  Then, the 
placeholders in the template ({article_title} and {article_text}) are replaced with the actual title and text of 
the article using the .format() method. 

"""
# prepare template for prompt
template = """
As an advanced AI, you've been tasked to summarize online articles into bulleted points. Here are a few examples of how you've done this in the past:

Example 1:
Original Article: 'The Effects of Climate Change
Summary:
- Climate change is causing a rise in global temperatures.
- This leads to melting ice caps and rising sea levels.
- Resulting in more frequent and severe weather conditions.

Example 2:
Original Article: 'The Evolution of Artificial Intelligence
Summary:
- Artificial Intelligence (AI) has developed significantly over the past decade.
- AI is now used in multiple fields such as healthcare, finance, and transportation.
- The future of AI is promising but requires careful regulation.

Now, here's the article you need to summarize:

==================
Title: {article_title}

{article_text}
==================

Please provide a summarized version of the article in a bulleted list format.
"""

# Format the Prompt
prompt = template.format(article_title=article.title, article_text=article.text)

messages = [HumanMessage(content=prompt)]

print(messages)

"""
Next step is to use ChatOpenAI class to load the GPT-4 model for generating the summary. 
Then, the formatted prompt is passed to the language model as the input/prompt. The ChatOpenAI 
class's chat instance takes a HumanMessage list as an input argument.

The key takeaway here is the use of a few-shot learning style in the prompt. This provides the 
model with examples of how it should perform the task, which guides it to generate a bulleted 
list summarizing the article. By modifying the prompt and the examples, you can adjust the model's 
output to meet various requirements and ensure the model follows a specified format, tone, style, etc.

"""
from langchain.chat_models import ChatOpenAI

# load the model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

# generate summary
summary = chat(messages)
print(summary.content)


"""
Now, let’s improve the previous section by using Output Parsers. The Pydantic output parser in LangChain 
offers a flexible way to shape the outputs from language models according to pre-defined schemas.  When 
used alongside prompt templates, it enables more structured interactions with language models, making it 
easier to extract and work with the information provided by the model.

The prompt template includes the format instructions from our parser, which guide the language model to 
produce the output in the desired structured format. The idea is to demonstrate how you could use 
PydanticOutputParser class to receive the output as a type List that holds each bullet point instead of a 
string. The advantage of having a list is the possibility to loop through the results or index a specific item.

As mentioned before, the PydanticOutputParser wrapper is used to create a parser that will parse the 
output from the string into a data structure. The custom ArticleSummary class, which inherits the Pydantic 
package’s BaseModel class, will be used to parse the model’s output.

We defined the schema to present a title along with a summary variable that represents a list of strings using 
the Field object. The description argument will describe what each variable must represent and help the model to 
achieve it. Our custom class also includes a validator function to ensure that the generated output contains at least 
three bullet points.
"""

from langchain.output_parsers import PydanticOutputParser
from pydantic import validator
from pydantic import BaseModel, Field
from typing import List


# create output parser class
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    # validating whether the generated summary has at least three lines
    @validator("summary", allow_reuse=True)
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines


# set up output parser
parser = PydanticOutputParser(pydantic_object=ArticleSummary)


"""
The next step involves creating a template for the input prompt that instructs the language 
model to summarize the news article into bullet points. This template is used to instantiate 
a PromptTemplate object, which is responsible for correctly formatting the prompts that are 
sent to the language model. The PromptTemplate uses our custom parser to format the prompt 
sent to the language model using the .get_format_instructions() method, which will include 
additional instructions on how the output should be shaped.
"""
from langchain.prompts import PromptTemplate


# create prompt template
# notice that we are specifying the "partial_variables" parameter
template = """
You are a very good assistant that summarizes online articles.

Here's the article you want to summarize.

==================
Title: {article_title}

{article_text}
==================

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["article_title", "article_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Format the prompt using the article title and text obtained from scraping
formatted_prompt = prompt.format_prompt(
    article_title=article_title, article_text=article_text
)


"""
Lastly, the GPT-3 model with the temperature set to 0.0  is initialized, which means the output will be 
deterministic, favoring the most likely outcome over randomness/creativity. The parser object then converts 
the string output from the model to a defined schema using the .parse() method.
"""
from langchain.llms import OpenAI

# instantiate model class
model = OpenAI(model_name="text-davinci-003", temperature=0.0)

# Use the model to generate a summary
output = model(formatted_prompt.to_string())

# Parse the output into the Pydantic model
parsed_output = parser.parse(output)
print(parsed_output)


"""
The Pydantic output parser is a powerful method for molding and structuring the output from language models. 
It uses the Pydantic library, known for its data validation capabilities, to define and enforce data schemas 
for the model's output.

This is a recap of what we did:

We defined a Pydantic data structure named ArticleSummary. This model serves as a blueprint for the desired 
structure of the generated article summary. It comprises fields for the title and the summary, which is expected 
to be a list of strings representing bullet points. Importantly, we incorporate a validator within this model to 
ensure the summary comprises at least three points, thereby maintaining a certain level of detail in the summarization.
We then instantiate a parser object using our ArticleSummary class. This parser plays a crucial role in ensuring 
the output generated by the language model aligns with the defined structures of our custom schema.

To direct the language model's output, we create the prompt template. The template instructs the model to act as 
an assistant that summarizes online articles by incorporating the parser object.

So, output parsers enable us to specify the desired format of the model's output, making extracting meaningful 
information from the model's responses easier.
"""

import os
from dotenv import load_dotenv
load_dotenv()

""" 
Note: install ffmpeg with:
sudo apt install ffmpeg
"""

import yt_dlp

def download_mp4_from_youtube(url):
    # Set the options for the download
    filename = 'lecuninterview.mp4'
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': filename,
        'quiet': True,
    }

    # Download the video file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)

url = "https://www.youtube.com/watch?v=mBjPyte2ZZo"
if os.path.exists("lecuninterview.mp4"):
    print("Video file already exists")
else:
    print("Downloading video")
    download_mp4_from_youtube(url)

"""
Whisper is a cutting-edge, automatic speech recognition system developed by OpenAI. Boasting 
state-of-the-art capabilities, Whisper has been trained on an impressive 680,000 hours of 
multilingual and multitasking supervised data sourced from the web.  This vast and varied 
dataset enhances the system's robustness, enabling it to handle accents, background noise, and 
technical language easily. OpenAI has released the models and codes to provide a solid foundation 
for creating valuable applications harnessing the power of speech recognition.

The whisper package that we installed earlier provides the .load_model() method to download the 
model and transcribe a video file. Multiple different models are available: tiny, base, small, medium, 
and large. Each one of them has tradeoffs between accuracy and speed. We will use the 'base' model 
for this tutorial.
"""

import whisper

if os.path.exists("lecuninterview_transcribed.txt"):
    print("Transcription already exists")
else:
    model = whisper.load_model("base")
    result = model.transcribe("lecuninterview.mp4")
    print("=== Transcribed video text ===")
    print(result['text'])

    # We’ve got the result in the form of a raw text and it is possible to save it to a text file.
    with open ('lecuninterview_transcribed.txt', 'w') as file:  
        file.write(result['text'])

# Now summarize with LangChain
from langchain import OpenAI, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(model_name="text-davinci-003", temperature=0)


"""
This code creates an instance of the RecursiveCharacterTextSplitter
class, which is responsible for splitting input text into smaller chunks. 
It is configured with a chunk_size of 1000 characters, no chunk_overlap, 
and uses spaces, commas, and newline characters as separators. This ensures 
that the input text is broken down into manageable pieces, allowing for 
efficient processing by the language model.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)


"""
Each Document object is initialized with the content of a chunk from the texts list. 
The [:4] slice notation indicates that only the first four chunks will be used to 
create the Document objects. 

The textwrap library in Python provides a convenient way to wrap and format plain text 
by adjusting line breaks in an input paragraph. It is particularly useful when displaying 
text within a limited width, such as in console outputs, emails, or other formatted text 
displays. The library includes convenience functions like wrap, fill, and shorten, as well 
as the TextWrapper class that handles most of the work. If you’re curious, I encourage you to 
follow this link and find out more, as there are other functions in the textwrap library that can be 
useful depending on your needs.
https://docs.python.org/3/library/textwrap.html
"""
from langchain.docstore.document import Document

with open('lecuninterview_transcribed.txt') as f:
    text = f.read()

texts = text_splitter.split_text(text)
docs = [Document(page_content=t) for t in texts[:4]]


from langchain.chains.summarize import load_summarize_chain
import textwrap

chain = load_summarize_chain(llm, chain_type="map_reduce")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print("=== Summary text from map reduce ===")
print(wrapped_text)


"""
With the following line of code, we can see the prompt template that is used with 
the map_reduce technique. Now we’re changing the prompt and using another summarization 
method:
"""
print("=== Prompt template ===")
print(chain.llm_chain.prompt.template)


"""
The "stuff" approach is the simplest and most naive one, in which all the text from the transcribed 
video is used in a single prompt. This method may raise exceptions if all text is longer than the 
available context size of the LLM and may not be the most efficient way to handle large amounts of text. 

We’re going to experiment with the prompt below. This prompt will output the summary as bullet points.
"""
prompt_template = """Write a concise bullet point summary of the following:


{text}


CONSCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, 
                        input_variables=["text"])


"""
Also, we initialized the summarization chain using the stuff as chain_type and the prompt above.
"""
chain = load_summarize_chain(llm, 
                             chain_type="stuff", 
                             prompt=BULLET_POINT_PROMPT)

output_summary = chain.run(docs)

wrapped_text = textwrap.fill(output_summary, 
                             width=1000,
                             break_long_words=False,
                             replace_whitespace=False)
print("=== Summary text from stuff ===")
print(wrapped_text)


"""
Great job! By utilizing the provided prompt and implementing the appropriate summarization 
techniques, we've successfully obtained concise bullet-point summaries of the conversation.

In LangChain we have the flexibility to create custom prompts tailored to specific needs. 
For instance, if you want the summarization output in French, you can easily construct a 
prompt that guides the language model to generate a summary in the desired language.

The 'refine' summarization chain is a method for generating more accurate and context-aware 
summaries. This chain type is designed to iteratively refine the summary by providing additional 
context when needed. That means: it generates the summary of the first chunk. Then, for each 
successive chunk, the work-in-progress summary is integrated with new info from the new chunk.

The 'refine' summarization chain in LangChain provides a flexible and iterative approach to 
generating summaries, allowing you to customize prompts and provide additional context for 
refining the output. This method can result in more accurate and context-aware summaries 
compared to other chain types like 'stuff' and 'map_reduce'.
"""
chain = load_summarize_chain(llm, chain_type="refine")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print("=== Summary text from refine ===")
print(wrapped_text)


"""
Adding transcripts to Deep Lake
This method can be extremely useful when you have more data. Let’s see how we can improve our expariment 
by adding multiple URLs, store them in Deep Lake database and retrieve information using QA chain.

First, we need to modify the script for video downloading slightly, so it can work with a list of URLs.
"""

print("=== Batch videos ===")
import yt_dlp

def download_mp4_from_youtube(urls, job_id):
    # This will hold the titles and authors of each downloaded video
    video_info = []

    for i, url in enumerate(urls):
        # Set the options for the download
        file_temp = f'./{job_id}_{i}.mp4'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': file_temp,
            'quiet': True,
        }

        # Download the video file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', "")
            author = result.get('uploader', "")

        # Add the title and author to our list
        video_info.append((file_temp, title, author))

    return video_info

if os.path.exists("1_0.mp4") and os.path.exists("1_1.mp4"):
    print("Video files already exist")
    video_details = [("1_0.mp4"), ("1_1.mp4")]
    print(video_details)
else:
    urls=["https://www.youtube.com/watch?v=mBjPyte2ZZo&t=78s",
        "https://www.youtube.com/watch?v=cjs7QKJNVYM",]
    video_details = download_mp4_from_youtube(urls, 1)
    print(video_details)


"""
And transcribe the videos using Whisper as we previously saw and save the results in a text file.
"""
print("=== Batch videos transcription ===")
import whisper

# load the model
model = whisper.load_model("base")

# iterate through each video and transcribe
results = []
for video in video_details:
    print("Video:", video)
    result = model.transcribe(video)
    results.append( result['text'] )
    print(f"Transcription for {video[0]}:\n{result['text']}\n")

with open ('batch_videos_transcribed.txt', 'w') as file:
    combined = ' '.join(results)  
    file.write(combined)


print("=== Store transcripts in deep lake ===")

"""
Then, load the texts from the file and use the text splitter to split the text to chunks 
with zero overlap before we store them in Deep Lake.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the texts
with open('batch_videos_transcribed.txt') as f:
    text = f.read()
texts = text_splitter.split_text(text)

# Split the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
texts = text_splitter.split_text(text)


"""
Similarly, as before we’ll pack all the chunks into a Documents:
"""
from langchain.docstore.document import Document

docs = [Document(page_content=t) for t in texts[:4]]

from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_course_youtube_summarizer"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
db.add_documents(docs)


"""
In order to retrieve the information from the database, we’d have to construct a retriever object.
"""
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4


"""
The distance metric determines how the Retriever measures "distance" or similarity between different 
data points in the database. By setting distance_metric to 'cos', the Retriever will use cosine similarity 
as its distance metric. Cosine similarity is a measure of similarity between two non-zero vectors of an 
inner product space that measures the cosine of the angle between them. It's often used in information 
retrieval to measure the similarity between documents or pieces of text. Also, by setting 'k' to 4, 
the Retriever will return the 4 most similar or closest results according to the distance metric when 
a search is performed.

We can construct and use a custom prompt template with the QA chain. The RetrievalQA chain is useful to 
query similiar contents from databse and use the returned records as context to answer questions. The 
custom prompt ability gives us the flexibility to define custom tasks like retrieving the documents and 
summaizing the results in a bullet-point style.
"""
from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Summarized answer in bullter points:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


"""
Lastly, we can use the chain_type_kwargs argument to define the custom prompt and for chain 
type the ‘stuff’  variation was picked. You can perform and test other types as well, as seen 
previously.

Of course, you can always tweak the prompt to get the desired result, experiment more with 
modified prompts using different types of chains and find the most suitable combination. Ultimately, 
the choice of strategy depends on the specific needs and constraints of your project. 

"""
print("=== Retrieve from deep lake ===")
from langchain.chains import RetrievalQA

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 chain_type_kwargs=chain_type_kwargs)

print(qa.run("Summarize the mentions of google according to their AI program"))


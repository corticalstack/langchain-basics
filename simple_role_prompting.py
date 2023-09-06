from dotenv import load_dotenv

load_dotenv()

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

"""
This is a good prompt for several reasons:

Clear instructions: The prompt is phrased as a clear request for help in generating a song title, 
and it specifies the context: "As a futuristic robot band conductor." This helps the LLM understand 
that the desired output should be a song title related to a futuristic scenario.

Specificity: The prompt asks for a song title that relates to a specific theme and a specific year, 
"{theme} in the year {year}." This provides enough context for the LLM to generate a relevant and 
creative output. The prompt can be adapted for different themes and years by using input variables, 
making it versatile and reusable.

Open-ended creativity: The prompt allows for open-ended creativity, as it doesn't limit the LLM to 
a particular format or style for the song title. The LLM can generate a diverse range of song titles 
based on the given theme and year.

Focus on the task: The prompt is focused solely on generating a song title, making it easier for the 
LLM to provide a suitable output without getting sidetracked by unrelated topics.

These elements help the LLM understand the user's intention and generate a suitable response.
"""

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""
prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template,
)

# Create the LLMChain for the prompt
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# Input data for the prompt
input_data = {"theme": "interstellar travel", "year": "3030"}

# Create LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the LLMChain to get the AI-generated song title
response = chain.run(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)

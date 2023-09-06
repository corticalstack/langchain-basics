from dotenv import load_dotenv

load_dotenv()

"""
This prompt:

Provides a clear context in the prefix: The prompt states that the AI is a life coach 
providing insightful and practical advice. This context helps guide the AI's responses 
and ensures they align with the intended purpose.

Uses examples that demonstrate the AI's role and the type of responses it generates: By 
providing relevant examples, the AI can better understand the style and tone of the responses 
it should produce. These examples serve as a reference for the AI to generate similar responses 
that are consistent with the given context.

Separates examples and the actual query: This allows the AI to understand the format it should 
follow, ensuring a clear distinction between example conversations and the user's input. This 
separation helps the AI to focus on the current query and respond accordingly.

Includes a clear suffix that indicates where the user's input goes and where the AI should provide 
its response: The suffix acts as a cue for the AI, showing where the user's query ends and the AI's 
response should begin. This structure helps maintain a clear and consistent format for the generated 
responses.

By using this well-structured prompt, the AI can understand its role, the context, and the expected 
response format, leading to more accurate and useful outputs.
"""

from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain
from langchain.llms import OpenAI

# Initialize LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0)

examples = [
    {
        "query": "What's the secret to happiness?",
        "answer": "Finding balance in life and learning to enjoy the small moments.",
    },
    {
        "query": "How can I become more productive?",
        "answer": "Try prioritizing tasks, setting goals, and maintaining a healthy work-life balance.",
    },
]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"], template=example_template
)

prefix = """The following are excerpts from conversations with an AI
life coach. The assistant provides insightful and practical advice to the users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n",
)

# Create the LLMChain for the few-shot prompt template
chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

# Define the user query
user_query = "What are some tips for improving communication skills?"

# Run the LLMChain for the user query
response = chain.run({"query": user_query})

print("User Query:", user_query)
print("AI Response:", response)

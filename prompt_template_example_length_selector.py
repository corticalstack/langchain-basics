import os
from dotenv import load_dotenv

from langchain import LLMChain, FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI
from langchain import LLMChain, FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

llm = OpenAI(model_name="text-davinci-003", temperature=0)

"""
Prompt Templates are essential for generating effective prompts for large language models, 
providing a structured and consistent format that maximizes accuracy and relevance. 
Integrating dynamic prompts enhances context understanding, flexibility, and results, making 
them a valuable asset for language model development.
"""

examples = [
    {
        "query": "How do you feel today?",
        "answer": "As an AI, I don't have feelings, but I've got jokes!"
    }, {
        "query": "What is the speed of light?",
        "answer": "Fast enough to make a round trip around Earth 7.5 times in one second!"
    }, {
        "query": "What is a quantum computer?",
        "answer": "A magical box that harnesses the power of subatomic particles to solve complex problems."
    }, {
        "query": "Who invented the telephone?",
        "answer": "Alexander Graham Bell, the original 'ringmaster'."
    }, {
        "query": "What programming language is best for AI development?",
        "answer": "Python, because it's the only snake that won't bite."
    }, {
        "query": "What is the capital of France?",
        "answer": "Paris, the city of love and baguettes."
    }, {
        "query": "What is photosynthesis?",
        "answer": "A plant's way of saying 'I'll turn this sunlight into food. You're welcome, Earth.'"
    }, {
        "query": "What is the tallest mountain on Earth?",
        "answer": "Mount Everest, Earth's most impressive bump."
    }, {
        "query": "What is the most abundant element in the universe?",
        "answer": "Hydrogen, the basic building block of cosmic smoothies."
    }, {
        "query": "What is the largest mammal on Earth?",
        "answer": "The blue whale, the original heavyweight champion of the world."
    }, {
        "query": "What is the fastest land animal?",
        "answer": "The cheetah, the ultimate sprinter of the animal kingdom."
    }, {
        "query": "What is the square root of 144?",
        "answer": "12, the number of eggs you need for a really big omelette."
    }, {
        "query": "What is the average temperature on Mars?",
        "answer": "Cold enough to make a Martian wish for a sweater and a hot cocoa."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative and funny responses to users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """


"""
By employing the LengthBasedExampleSelector, the code dynamically selects and includes 
examples based on their length, ensuring that the final prompt stays within the desired 
token limit. The selector is employed to initialize a dynamic_prompt_template:
"""
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100  
)


"""
So, the dynamic_prompt_template utilizes the example_selector instead of a fixed list of examples. 
This allows the FewShotPromptTemplate to adjust the number of included examples based on the length 
of the input query. By doing so, it optimizes the use of the available context window and ensures 
that the language model receives an appropriate amount of contextual information.
"""
dynamic_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)

# Create the LLMChain for the dynamic_prompt_template
chain = LLMChain(llm=llm, prompt=dynamic_prompt_template)

# Run the LLMChain with input_data
input_data = {"query": "Who invented the telephone?"}
response = chain.run(input_data)

print(response)


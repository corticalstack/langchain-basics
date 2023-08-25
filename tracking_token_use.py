from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()

llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2)

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)
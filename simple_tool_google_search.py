from dotenv import load_dotenv

load_dotenv()

# As a standalone utility:
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()
print(search.results("What is the capital of Spain?", 3))

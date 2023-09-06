from dotenv import load_dotenv

load_dotenv()

from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

wolfram = WolframAlphaAPIWrapper()
result = wolfram.run("What is 2x+5 = -3x + 7?")
print(result)  # Output: 'x = 2/5'

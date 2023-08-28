"""
A sequential chain that concatenates multiple chains into one.
The SimpleSequentialChain will start running each chain from the first index and pass its response to the next one in the list.
"""
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two])

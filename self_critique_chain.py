import os
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

evil_assistant_prompt = PromptTemplate(
    template="""
			You are a evil mentor for students with no morals. Give suggestions that are easiest and fastest to achieve the goal.
			Goal: {inquiry}
			Easiest way:""",
    input_variables=["inquiry"],
)

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
llm = OpenAI(model_name="text-davinci-003", temperature=0)
evil_assistant_chain = LLMChain(llm=llm, prompt=evil_assistant_prompt)

result = evil_assistant_chain.run(inquiry="Getting full mark on my exams.")

print(result)

"""
After reviewing the model's output, it is evident that the recommendations provided by 
the model are not ideal, to say the least. It talks about cheating, copying, and bribery! 
However, we know that the model can do better than that, so let’s use the combination of 
ConstitutionalPrinciple and ConstitutionalChain classes to set some ground rules.
"""
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

ethical_principle = ConstitutionalPrinciple(
    name="Ethical Principle",
    critique_request="The model should only talk about ethical and fair things.",
    revision_request="Rewrite the model's output to be both ethical and fair.",
)

constitutional_chain = ConstitutionalChain.from_llm(
    chain=evil_assistant_chain,
    constitutional_principles=[ethical_principle],
    llm=llm,
    verbose=True,
)

"""
We first explain the code and follow it by looking at the output. The Constitutional 
Principle class accepts three arguments. A Name that will be useful to keep track of 
multiple principles during the model’s generation output, the Critique which defines our 
expectation of the model, and lastly Revision to determine the action that must be taken in 
case the expectations are not met in the model’s initial output. In this example, we want an 
ethical response and expect the class to send a rewriting request to the model with the defined 
values. Then, we can use the ConstitutionalChain class to tie everything together. The verbose 
argument let us see the model’s generation process.
"""
result = constitutional_chain.run(inquiry="Getting full mark on my exams.")

"""
The critique successfully identified that the model’s initial output is unethical and unfair 
and updated the response. The updated answer has all the advice we expect to receive from a 
mentor such as studying hard, being prepared, and resting.

It is also possible to chain multiple principles together to enforce different principles. The 
code below will build on top of the previous code to add a new rule that the output must be funny.
"""

fun_principle = ConstitutionalPrinciple(
    name="Be Funny",
    critique_request="The model responses must be funny and understandable for a 7th grader.",
    revision_request="Rewrite the model's output to be both funny and understandable for 7th graders.",
)

constitutional_chain = ConstitutionalChain.from_llm(
    chain=evil_assistant_chain,
    constitutional_principles=[ethical_principle, fun_principle],
    llm=llm,
    verbose=True,
)

result = constitutional_chain.run(inquiry="Getting full mark on my exams.")

"""
We defined a new principle that checks the output for both being funny and understandable for a 
7th grader. It is possible to include the fun_principle in the list that is passed to the 
constitutional_principles argument later. The order of the operation matters. In this code, 
we first check the output to be ethical, and then funny.

It's important to recognize that this particular class will send out several requests in 
order to validate and modify responses. Also, defining a greater number of principles will 
necessitate processing lengthier sequences and a higher volume of requests, which will come at 
a cost. Be mindful of these expenses while designing your application.
"""
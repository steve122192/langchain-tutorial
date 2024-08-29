import os
from dotenv import load_dotenv
load_dotenv()

from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

prompt_template = """
        given the Job title  about a person, I want you to create:
        1. a description of what their day at work might look like
        2. How much money they might make
        Job Title: {title}
    """
prompt = PromptTemplate(input_variables=['title'], template=prompt_template)


chain = prompt | llm



if __name__ == "__main__":
    print(chain.invoke(input={"title": 'Zookeeper'}))
    #breakpoint()
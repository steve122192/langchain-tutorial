import os
from dotenv import load_dotenv
load_dotenv()

from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Tuple

llm = ChatOpenAI(temperature=0)

class JobDescription(BaseModel):
    description: str = Field(description='Description of day to day activities')
    salary: Tuple[str,str] =  Field(description="potential salary range as integer with no symbols or commas")

parser = PydanticOutputParser(pydantic_object=JobDescription)

prompt_template = """
        given the Job title  about a person, I want you to create:
        1. a description of what their day at work might look like
        2. How much money they might make
        Job Title: {title}

        \n{format_instructions}
    """
prompt = PromptTemplate(input_variables=['title'], template=prompt_template,
                        partial_variables={"format_instructions": parser.get_format_instructions()})


chain_output = prompt | llm | parser



if __name__ == "__main__":
    print(chain_output.invoke(input={"title": 'Zookeeper'}))
    #breakpoint()
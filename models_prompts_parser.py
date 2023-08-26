import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

# Environment Variables
load_dotenv(find_dotenv())

# Setting Up OPENAI Config
openai.api_type="azure"
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version='2023-05-15'

class InfoExtractor:
    def __init__(self, template):
        self.prompt_template = ChatPromptTemplate.from_template(template)
    
    def get_schemas(self, formats):
        response_schemas = []
        for format in formats:
            response_schemas.append(ResponseSchema(name=format['name'], description=format['description']))
        
        return response_schemas


    def get_info(self, review, response_format, temperature=0):
        chat = ChatOpenAI(temperature=temperature, model_kwargs = {"engine" : "GPT3-5"})
        response_schemas = self.get_schemas(response_format)
        message = self.prompt_template.format_messages(text=review)
        response = chat(message)
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        return output_parser.parse(response.content)
    
    

if __name__ == '__main__':
    review_template = """\
    For the following text, extract the following information:

    gift: Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown.

    delivery_days: How many days did it take for the product \
    to arrive? If this information is not found, output -1.

    price_value: Extract any sentences about the value or price,\
    and output them as a comma separated Python list.

    Format the output as JSON with the following keys:
    gift
    delivery_days
    price_value

    text: {text}
    """

    customer_review = """\
    This leaf blower is pretty amazing.  It has four settings:\
    candle blower, gentle breeze, windy city, and tornado. \
    It arrived in two days, just in time for my wife's \
    anniversary present. \
    I think my wife liked it so much she was speechless. \
    So far I've been the only one using it, and I've been \
    using it every other morning to clear the leaves on our lawn. \
    It's slightly more expensive than the other leaf blowers \
    out there, but I think it's worth it for the extra features.
    """

    response_format = [
        {
            "name": "gift",
            "description" : "Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown."
        },
        {
            "name": "delivery_days",
            "description": "How many days did it take for the product to arrive? If this  information is not found, output -1."
        },
        {
            "name": "price_value",
            "description": "Extract anysentences about the value or price, and output them as a comma separated Python list."
        }
    ]
    
    info_extractor = InfoExtractor(review_template)
    print(info_extractor.get_info(customer_review, response_format))
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)
def initialise_llama3():
    try:
        # Create chatbot prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are my personal assistant"),
                ("user", "Question: {question}")
            ]
        )

        # Initialize OpenAI LLM and output parser
        llm = Ollama(model="llama3")
        output_parser = StrOutputParser()

        # Create chain
        chain = prompt | llm | output_parser
        return chain
    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        raise

# Initialize chatbot
chatbot_pipeline = initialise_llama3()

def main():
    query_input = "Who is God"
    try:
        response = chatbot_pipeline.invoke({'question': query_input})
        print(response)
    except Exception as e:
        logging.error(f"Error during chatbot invocation: {e}")

if __name__ == '__main__':
    main()
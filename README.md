# 🚀 LangChain Basics

A comprehensive collection of examples and utilities for working with LangChain, a framework for building applications powered by large language models (LLMs).

## 📚 Description

This repository serves as a practical guide and reference for developers looking to leverage the LangChain framework. It contains numerous examples demonstrating various LangChain components and capabilities, from simple chains to complex agents and applications.

The examples cover a wide range of use cases, including:
- Building conversational agents
- Creating document-based question answering systems
- Implementing memory for contextual conversations
- Working with different output formats
- Processing and analyzing various data sources
- Building complete applications

## ✨ Features

- **Chains**: Examples of different chain types and their usage
  - [Simple Chain](simple_chain.py) - Basic LLMChain example
  - [Sequential Chain](sequential_chain.py) - Chaining multiple LLMs together
  - [Custom Chain](custom_chain.py) - Creating custom chain implementations

- **Agents**: Autonomous agents that can use tools to accomplish tasks
  - [Google Search Agent](simple_agent_google_search.py) - Agent that can search the web
  - [Wikipedia Agent](simple_tool_wikipedia.py) - Agent that can query Wikipedia
  - [WolframAlpha Agent](simple_tool_wolframalpha.py) - Agent for mathematical computations
  - [AutoGPT](simple_autogpt.py) - Implementation of AutoGPT using LangChain
  - [BabyAGI](simple_babyagi.py) - Implementation of BabyAGI using LangChain

- **Memory**: Different memory implementations for conversational contexts
  - [Conversation Buffer Memory](conversation_memory_buffer.py) - Simple memory that stores all messages
  - [Conversation Buffer Window Memory](conversation_buffer_window_memory.py) - Memory with a sliding window
  - [Conversation Summary Memory](conversation_summary_memory.py) - Memory that summarizes past conversations

- **Prompts**: Various prompting techniques and templates
  - [Simple Prompt](simple_prompt.py) - Basic prompt usage
  - [Prompt Templates](simple_prompt_template.py) - Using templates for consistent prompting
  - [Few-Shot Prompting](simple_few_shot_prompting.py) - Learning from examples
  - [Role Prompting](simple_role_prompting.py) - Assigning roles to the LLM

- **Output Parsers**: Structuring and formatting LLM outputs
  - [Structured Output Parser](output_parser_structured.py) - Parsing outputs into structured formats
  - [Pydantic Output Parser](output_parser_pydantic.py) - Using Pydantic for type validation
  - [Retry Output Parser](output_retry_output_parser.py) - Handling parsing failures

- **Text Splitters**: Dividing text into manageable chunks
  - [Character Text Splitter](text_splitter_character.py) - Splitting by character count
  - [Recursive Text Splitter](text_splitter_recursive.py) - Intelligent recursive splitting
  - [Token Text Splitter](text_splitter_token.py) - Splitting based on token count

- **Document Processing**: Working with various document types
  - [DeepLake Document Loader](simple_deeplake_document_loader.py) - Loading documents into DeepLake
  - [DeepLake Document Retriever](simple_deeplake_document_retriever.py) - Retrieving documents from DeepLake

- **Applications**: Complete applications built with LangChain
  - [Chat with Any Data](chat-with-any-data/) - A Streamlit app for chatting with your data
  - [LLM Bot](llm-bot/) - A chatbot implementation using LangChain

## 🔧 Prerequisites

- Python 3.8+
- OpenAI API key (for most examples)
- Various API keys depending on the example (Google, WolframAlpha, etc.)

## 🛠️ Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/langchain-basics.git
cd langchain-basics
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
WOLFRAM_ALPHA_APPID=your_wolfram_alpha_appid
ACTIVELOOP_TOKEN=your_activeloop_token
ACTIVELOOP_ORG_NAME=your_activeloop_org_name
```

## 🚀 Usage

Each Python file in the repository is a standalone example that demonstrates a specific LangChain feature or concept. To run an example, simply execute the Python file:

```bash
python simple_chain.py
```

For the Streamlit applications, navigate to the application directory and run:

```bash
cd chat-with-any-data
streamlit run app.py
```

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [OpenAI API Documentation](https://platform.openai.com/docs/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [DeepLake Documentation](https://docs.activeloop.ai/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

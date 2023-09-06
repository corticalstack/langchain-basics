import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import io
import re
import sys
from typing import Any, Callable

from langchain.vectorstores import DeepLake
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


def capture_and_display_output(func: Callable[..., Any], args, **kwargs) -> Any:
    # Capture the standard output
    original_stdout = sys.stdout
    sys.stdout = output_catcher = io.StringIO()

    # Run the given function and capture its output
    response = func(args, **kwargs)

    # Reset the standard output to its original value
    sys.stdout = original_stdout

    # Clean the captured output
    output_text = output_catcher.getvalue()
    clean_text = re.sub(r"\x1b[.?[@-~]", "", output_text)

    # Custom CSS for the response box
    st.markdown(
        """
    <style>
        .response-value {
            border: 2px solid #6c757d;
            border-radius: 5px;
            padding: 20px;
            background-color: #f8f9fa;
            color: #3d3d3d;
            font-size: 20px;  # Change this value to adjust the text size
            font-family: monospace;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create an expander titled "See Verbose"
    with st.expander("See Langchain Thought Process"):
        # Display the cleaned text in Streamlit as code
        st.code(clean_text)

    return response


def data_lake():
    my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
    my_activeloop_dataset_name = "langchain_knowledgebase"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    embeddings = CohereEmbeddings(model="embed-english-v2.0")

    dbs = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings)
    retriever = dbs.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    compressor = CohereRerank(model="rerank-english-v2.0", top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return dbs, compression_retriever, retriever


def memory():
    memory = ConversationBufferWindowMemory(
        k=3, memory_key="chat_history", return_messages=True, output_key="answer"
    )
    return memory


class App:
    def __init__(self):
        st.set_page_config(
            page_title="Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="auto",
        )
        if "dbs" not in st.session_state:
            (
                st.session_state["dbs"],
                st.session_state["compression_retriever"],
                st.session_state["retriever"],
            ) = data_lake()
        if "user_prompt" not in st.session_state:
            st.session_state["user_prompt"] = ""
        if "memory" not in st.session_state:
            st.session_state["memory"] = memory()
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "full_response" not in st.session_state:
            st.session_state["full_response"] = ""
        if "llm" not in st.session_state:
            optional_params = {
                "top_p": 0.95,
            }
            st.session_state["llm"] = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                model_kwargs=optional_params,
                temperature=0,
                streaming=True,
                verbose=True,
                max_tokens=1500,
            )
        if "qa" not in st.session_state:
            st.session_state["qa"] = ConversationalRetrievalChain.from_llm(
                llm=st.session_state["llm"],
                retriever=st.session_state["compression_retriever"],
                memory=st.session_state["memory"],
                verbose=True,
                chain_type="stuff",
                return_source_documents=True,
            )

    def main(self):
        with st.form("Ask Me"):
            st.session_state.user_prompt = st.text_input(
                label="Natural Language Query", help=("Enter query")
            )

            submitted = st.form_submit_button("Generate")
            if submitted:
                # Add user message to chat history
                st.session_state.messages.append(
                    {"role": "user", "content": st.session_state.user_prompt}
                )

                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(st.session_state.user_prompt)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()

                    # Load the memory variables, which include the chat history
                    memory_variables = st.session_state["memory"].load_memory_variables(
                        {}
                    )

                    # Predict the AI's response in the conversation
                    with st.spinner("Searching course material"):
                        response = capture_and_display_output(
                            st.session_state["qa"],
                            (
                                {
                                    "question": st.session_state.user_prompt,
                                    "chat_history": memory_variables,
                                }
                            ),
                        )

                        # Display chat response
                        st.session_state["full_response"] += response["answer"]
                        message_placeholder.markdown(
                            st.session_state["full_response"] + "▌"
                        )
                        message_placeholder.markdown(st.session_state["full_response"])

                        # Display top 2 retrieved sources
                        source = response["source_documents"][0].metadata
                        source2 = response["source_documents"][1].metadata
                        with st.expander("See Resources"):
                            st.write(f"Title: {source['title'].split('·')[0].strip()}")
                            st.write(f"Source: {source['source']}")
                            st.write(
                                f"Relevance to Query: {source['relevance_score'] * 100}%"
                            )
                            st.write(f"Title: {source2['title'].split('·')[0].strip()}")
                            st.write(f"Source: {source2['source']}")
                            st.write(
                                f"Relevance to Query: {source2['relevance_score'] * 100}%"
                            )

        # Append message to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": st.session_state["full_response"]}
        )


if __name__ == "__main__":
    app = App()
    app.main()

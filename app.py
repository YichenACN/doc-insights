import os
import glob
import tempfile
import streamlit as st
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain.vectorstores import Chroma

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def document2vecdb(doc_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(doc_path)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    return db

def get_text():
    input_text = st.text_input("Ask your own question: ","", key="input")
    return input_text

def main():
    menu = ["Home", "Use Case 1", "Use Case 2", "Use Case 3"]
    choice = st.sidebar.selectbox("Select your role", menu)
    st.sidebar.markdown("----")
    model = st.sidebar.radio('Pick a model version', ('gpt-3.5-turbo-16k-0613', 'gpt-4-0613', 'PaLM2 (available soon)'))

    if choice == "Home":
        home()
    elif choice == "Use Case 1":
        UC1(model)
    elif choice == "Use Case 2":
        UC2(model)
    elif choice == "Use Case 3":
        UC3(model)

def home():
    st.title("Sustainability Accelerator")
    st.markdown("""Description""")
    st.markdown("----")
    if st.button("Clear Cache"):
        st.cache_data.clear()

def UC1(model):
    return

def UC2(model):
    return

def UC3(model):
    st.title("Document Insights")

    if "quick_insight" not in st.session_state:
        st.session_state.quick_insight = False

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
        
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    st.sidebar.markdown("----")
    # read subfolders in a give directory based on the actual directory level
    #foldernames_list = ["None"]
    #foldernames_list = foldernames_list + [os.path.basename(x) for x in glob.glob(f'{"data/*"}')]
    # create selectbox with the foldernames
    #chosen_folder = st.sidebar.selectbox(label="Choose a folder", options=foldernames_list)
    # set the full path to be able to refer to it
    #directory_path =  "data/" + chosen_folder
    uploaded_file = st.sidebar.file_uploader("Select your sustanibility annual report", accept_multiple_files=False, type="pdf")
    st.sidebar.markdown("----")
    if st.sidebar.button("Quick Insights"):
        st.session_state.quick_insight = True
    questions = [
        "What is the summary of their sustainability initiatives in 300 words",
        "What is their latest green gas emissions by scope(1,2,3)?",
        "Do they have a net zero target?",
        #"What are their goals to reduce carbon emissions? List the % reduction, by which year and what is their baseline year.",
        #"What sustainability initiatives do they have with a brief description of each?",
        #"What actions are they taking to reduce the carbon emission?",
        #"Do they have externally verified carbon reduction targets?"
    ]

    user_input = get_text()
    
    #if directory_path != "data/None":
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            fp = Path(tmp_file.name)
            fp.write_bytes(uploaded_file.getvalue())
            print(tmp_file.name)
        db = document2vecdb(tmp_file.name, 1000, 0)
        tmp_file.close()
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":15})
    
        qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(model_name=model, max_tokens=2000, temperature = 0.1), chain_type="stuff", retriever=retriever, return_source_documents=True, reduce_k_below_max_tokens=True) #

        if st.session_state.quick_insight:
            for q in questions:
                if q not in st.session_state.past:
                    result = qa({"question": q})
                    #with st.expander(q):
                    #    st.write(result['answer'])
                    st.session_state.past.append(q)
                    st.session_state.generated.append(result['answer'])

        if user_input:
            if user_input not in st.session_state.past:
                result = qa({"question": user_input})
                #with st.expander(user_input):
                #    st.write(result['answer'])
                st.session_state.past.append(user_input)
                st.session_state.generated.append(result['answer'])

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                with st.expander(st.session_state['past'][i]):
                    st.write(st.session_state['generated'][i])


if __name__ == "__main__":
    main()

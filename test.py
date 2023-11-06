import os
import openai
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader

_ = load_dotenv(find_dotenv())

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


openai.api_key  = os.environ['OPENAI_API_KEY']
llm_name = "gpt-3.5-turbo-0301"

# load documents
loader = PyPDFLoader("./internet.pdf")
# loader = TextLoader("./test.txt")
documents = loader.load()

print(len(documents))

# split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

print(len(docs))

# define embedding
embeddings = OpenAIEmbeddings()

# create vector database from data
# db = DocArrayInMemorySearch.from_documents(docs, embeddings)

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./docs/chroma/"
)

print(vectordb._collection.count())

llm = OpenAI(temperature=0)
# compressor = LLMChainExtractor.from_llm(llm)

# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=vectordb.as_retriever(search_type = "mmr")
# )

# question = "que es el internet"
# compressed_docs = compression_retriever.get_relevant_documents(question)
# pretty_print_docs(compressed_docs)

# Build prompt
# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=memory
)

question = "Cual es el tema principal del pdf?"
result = qa({"question": question})
print(result['answer'])

# question = "Que es el internet"

# docs = vectordb.similarity_search(question,k=3)

# len(docs)

# print(docs[5].page_content)

# define retriever
# retriever = db.as_retriever()


# print(memory)
# qa = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(model_name=llm_name, temperature=0),
#     retriever=retriever,
#     memory=memory
# )

# print(qa)

# question = "Que es el internet?"
# result = qa({"question": "hola"})

# result['answer']

# create a chatbot chain. Memory is managed externally.
# qa = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(model_name=llm_name, temperature=0), 
#     chain_type="stuff", 
#     retriever=retriever, 
#     return_source_documents=True,
#     return_generated_question=True,
# )

# print(qa({"question": "que es el internet?", "chat_history": ""}))
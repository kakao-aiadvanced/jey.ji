
import bs4 
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

def tavily_search(query):
    from tavily import TavilyClient
    tavily = TavilyClient(api_key='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')

    response = tavily.search(query, max_results=3)
    context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]

# You can easily get search result context based on any max tokens straight into your RAG.
# The response is a string of the context within the max_token limit.

    response_context = tavily.get_search_context(query, search_depth="advanced", max_tokens=500)

    # You can also get a simple answer to a question including relevant sources all with a simple function call:
    # You can use it for baseline
    response_qna = tavily.qna_search(query)
    return response_context, response_qna

### Index
def index():
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = vectorstore.as_retriever()

    return retriever

    
# The response is a string of the context within the max_token limit.

### Router
def router(question, llm):
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    system = """You are an expert at routing a user question to a vectorstore or web search.
    Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks.
    You do not need to be stringent with the keywords in the question related to these topics.
    Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
    Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}"),
        ]
    )

    question_router = prompt | llm | JsonOutputParser()

    # question = "llm agent memory"
    # question = "What is prompt?"
    
    return question_router.invoke({"question": question})

### Retrieval Grader
def retrieval_grader(question, llm, doc_txt):
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    system = """You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n document: {document} "),
        ]
    )

    retrieval_grader = prompt | llm | JsonOutputParser()
    
    return retrieval_grader.invoke({"question": question, "document": doc_txt})

### Generate
def generate(question, llm, doc_txt):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    system = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n context: {context} "),
        ]
    )

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": doc_txt, "question": question})
    print(generation)
    return generation

### Hallucination Grader
def hallucination_grader(llm, docs, generation):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    system = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "documents: {documents}\n\n answer: {generation} "),
        ]
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader.invoke({"documents": docs, "generation": generation})

### Answer Grader
def answer_grader(llm, question, generation):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
# Prompt
    system = """You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n answer: {generation} "),
        ]
    )

    answer_grader = prompt | llm | JsonOutputParser()
    return answer_grader.invoke({"question": question, "generation": generation})

# question = "llm agent memory"
# question = "What is prompt?"

input_query = input(" 1. llm agent memory\n 2. What is prompt?\n input 1 or 2 :")    
if input_query == "1":
    user_query = "llm agent memory"
elif input_query == "2":
    user_query = "What is prompt?"
else:
    user_query = input_query

llm = ChatOpenAI(model_name="gpt-4o-mini")
    

retriever = index()
docs = retriever.invoke(user_query)
document_text = docs[1].page_content


result = router(user_query, llm)
print(result)

if result.get("datasource") == "web_search":
    print("web_search")
    search_result = tavily_search(user_query)
    result = retrieval_grader(user_query, llm, search_result)
    if result.get("score") == "yes":
        print("success")
        print(generate(user_query, llm, search_result[0]) + "\nsource : web_search" )
    else:
        print("faild: not relevant")
        exit()

else:
    print("vectorstore")
    result = retrieval_grader(user_query, llm, document_text)
    if result.get("score") == "yes":
        print("success")
        generation = generate(user_query, llm, document_text)
        # print(generation + "\nsource : vectorstore" )

        hallucination_result = hallucination_grader(llm, docs, generation)
        if hallucination_result.get("score") == "yes":
            print("success")
            print(generate(user_query, llm, document_text) + "\nsource : vectorstore" )
            print(answer_grader(llm, user_query, generation))
        else:
            print("faild: hallucination")
    else:
        print("vectorstore faild")

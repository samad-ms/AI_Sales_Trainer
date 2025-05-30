from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

def answer_question_with_context(question, context_text):
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = [Document(page_content=context_text)]
    result = chain.run(input_documents=docs, question=question)
    return result

def answer_question_with_vector_store(question, vector_store):
    docs_and_scores = vector_store.similarity_search_with_score(question, k=3)
    context = "\n".join([doc.page_content for doc, _ in docs_and_scores])
    return answer_question_with_context(question, context)

import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pinecone import Pinecone
from langchain.vectorstores import Pinecone

#prompt
def generate_prompt(query): 
    results = vector_store.similarity_search(query, k=5)
    knowledge = "/n".join(doc.page_content for doc in results)
    prompt = f""" Use only this context given to you to answer the question. If you cannot answer the query using the knowledge base, clearly state so and end your answer. 
    
    Knowledge: {knowledge}

    Query: {query}
    """

    return prompt

try: 
    embeddings = AzureOpenAIEmbeddings(
            azure_deployment= "embedding",
            openai_api_version="2023-05-15",
        )
    model = AzureChatOpenAI(
        openai_api_version = "2023-05-15", 
        azure_deployment = "chat",
    )
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("knowledge-base")
except Exception as e: 
    print(f'An error occurred while initializing embeddings model and chat model: {e}')

vector_store = Pinecone(index, embeddings, text_key = "text")

while True:
    query = input("Human: ")
    if query.lower() == 'exit':
        break
    try:
        augmented_prompt = generate_prompt(query)
        response = model.invoke(augmented_prompt)
        print(f"Chat: {response.content}")
    except Exception as e:
        print(f"An error occurred while generating a response: {e}")

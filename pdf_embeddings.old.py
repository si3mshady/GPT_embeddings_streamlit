from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    try:
      st.set_page_config(page_title="GPT++", page_icon=None, layout="wide", initial_sidebar_state="expanded")

      load_dotenv()

      
      st.header("Elliotts Personal Tutor 💬")
      
      # upload file
      pdf = st.file_uploader("Upload your PDF", type="pdf")
      
      # extract the text
      if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
          text += page.extract_text()
          
        # split into chunks
        text_splitter = CharacterTextSplitter(
          separator="\n",
          chunk_size=1000,
          chunk_overlap=200,
          length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # show user input
        while True:
          user_question = st.text_input("Query your PDF for insight:")
          if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            
            try:
          
              if "aws" in response:
                st.code(response, language='python')
              else:
                st.write(response)
                 
            
            except Exception as e:
               st.write(response)
                
               

            # with get_openai_callback() as cb:
              # response = chain.run(input_documents=docs, question=user_question)
              # print(cb)
          
              # if "provider" in response:
              #   st.code(response, language='python')
              # else:
              #   st.write(response)

          
    except Exception as e:
       print(e)
   
      

if __name__ == '__main__':
    
    main()
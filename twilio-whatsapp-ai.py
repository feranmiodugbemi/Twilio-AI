from flask import Flask, request
import os
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import requests
import tempfile
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

app = Flask(__name__)

UPLOAD_FOLDER = 'pdfs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
pdf_exists = False
VectorStore = None
@app.route('/message', methods=['POST'])
def whatsapp():
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    response = None
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    client = Client(account_sid, auth_token)
    twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
    sender_phone_number = request.values.get('From', '')
    media_content_type = request.values.get('MediaContentType0')
    print(media_content_type)
    pdf_url = request.values.get('MediaUrl0')
    response = None
    if media_content_type == 'application/pdf':
        global pdf_exists, VectorStore
        pdf_exists = True
        response = requests.get(pdf_url)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
            pdf = PdfReader(temp_file_path)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len
            )
            chunks = text_splitter.split_text(text=text)
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            response = "Recieved, You can now ask your Questions"
    elif pdf_exists:
        question = request.values.get('Body')
        if pdf_exists:
            docs = VectorStore.similarity_search(query=question, k=3)
            llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.4)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=question)
            message = client.messages.create(
                body=answer,
                from_=twilio_phone_number,
                to=sender_phone_number
            )
            return str(message.sid)
        else:
            response = "No PDF file uploaded."
    else:
        print(media_content_type)
        response = "The media content type is not application/pdf"
    print(media_content_type)
    message = client.messages.create(
        body=response,
        from_=twilio_phone_number,
        to=sender_phone_number
    )

    return str(message.sid)

if __name__ == '__main__':
    app.run(debug=True)

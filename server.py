# This program is intended to establish a server that a client Raspberry-Pi running from the 
# JBAZZ repository: https://github.com/bwoodward523/JBAZZ-EmbeddedSystemsRobot.
# 
# Author: Brandon Woodward
# 
# This server will accept connections from a raspberry pi client.
# The server will then wait for communications containing bytes are received. 
# Then the server will create an audio file from those bytes and it will be 
# passed into an Speech-To-Text model to generate the text contents. 
# Then the server will pass the Text into an LLM with a carefully crafted prompt.
# The prompt will cause the LLM to generate an emotion, text, response and whether to shoot the nerf gun 
#
# For more details about any speciifc part reference the particular file regarding it. 
#
# Finally, the server will send the generated text, emotion, and whether to shoot to the client
# 
# Then the server will resume listening for the next incoming packet.
# 
# That means that until the client receives the response from the first message, any new messages will 
# not be received. Adding a message queue is a stretch goal that could allow for interesting behavior 
# like speech interruptions, this could include a timer or some indication that JBAZZ was cutoff and may make him angry.


import socket
import struct 
from llm import * 
from sst import *
from faster_whisper import WhisperModel



HOST = "0.0.0.0" #Temporary host to listen to all possible connections
PORT = 5555  

def recv_exact(sock, n):
    buffer = b''
    while len(buffer) < n:
        chunk = sock.recv(n - len(buffer))
        if not chunk:
            return None
        buffer += chunk
    return buffer


def send_message(sock, payload: bytes):
    header = struct.pack("!I", len(payload))
    sock.sendall(header)
    sock.sendall(payload)


def recv_message(sock):
    header = recv_exact(sock, 4)
    if header is None:
        return None

    length = struct.unpack("!I", header)[0]
    return recv_exact(sock, length)


def handle_client(conn, addr):
    print(f"Client connected: {addr}")
    #Check if ollama is active
    try:
        while True:
            payload = recv_message(conn)

            #Handle termination of client's connection. 
            if payload is None:
                print("Client disconnected. \nAwaiting new connection.")
                llm.reset()
                break

            # Decode ONLY after full payload received
            # request = payload
            # print("Received:", request)

                

            # Receive the audio and convert it to text
            client_text = "Hello world"
            if is_sst_online:
                client_text = convert_to_text(payload, whisper_model) #TODO: add payload
                print(client_text)
            else:
                print("No SST implemented.")

            # Pass the text into the LLM
            llm_text_output = "LLM Text output"
            if is_llm_online:
                llm_text_output = llm.ask(client_text)

                def validate_reply(text):
                    required = ['emotion:', '!@#$', 'text response:', 'shoot:']
                    return all(token in text for token in required)
                
                if not validate_reply(llm_text_output):
                    llm_text_output = 'emotion: "anger"!@#$ text response: "Formatting failure. Try again."!@#$ shoot: "False"'

            else:
                pass
        

            response = f"Processed:  {client_text} \n {llm_text_output}"
            send_message(conn, response.encode("utf-8"))

    except ConnectionResetError:
        print("Client crashed / reset connection.")

    finally:
        conn.close()


def run_server():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)

        print(f"Listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            handle_client(conn, addr)


if __name__ == "__main__":

    is_sst_online = False
    try:    
        whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
        is_sst_online = True

    except Exception as e:
        print(e) 
    
    is_llm_online = False
    try:
        llm = LLMContext()
        is_llm_online = True
    except Exception as e:
        print(e)
        

    run_server()
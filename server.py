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
import queue
import threading


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

from enum import Enum
#Class to indicate which state of outputting in the LLM we are in. 
class LLM_OUTPUT_STATE(Enum):
    NONE = -1
    EMOTION = 0
    CHARACTERS = 1
    SHOOT = 2
DELIMITER = '@#$'

#Thread for handling the data as it comes 
def consume_llm_stream(character_queue, conn):       
    
    def has_delimiter(s):
        #If it finds the delimiter. 
        #It should attempt to get the content to the left and right of the delimiter.
        #Return this content in a tuple pair. 
        #Refactor state code to use left side as message and the right side as the start of s. 
        if s.count("@#$") == 1:
            print(f"Delimter exists within s: {s}")

            #Extract left and right content of delimiter
            s = s.split("@#$")
            print(f"split array {s}")

            return (s[0], s[1])
        else:
            return None
        
    LLM_STATE = LLM_OUTPUT_STATE.EMOTION
    s = ''
    full_response = ''
    swords = []
    while True:
        chunk = character_queue.get()
        # print(f"Chunk we are reading: {chunk}")

        assert type(chunk) == str
        #Extend the s string with the new chunk streamed 
        s += chunk
        full_response += chunk
        if LLM_STATE == LLM_OUTPUT_STATE.EMOTION:
            d_result = has_delimiter(s)
            if d_result:
                print(f"Current state of s when emotion is ready: {s}")
                #Send the emotion to JBAZZ. We can remove the delimiter. 
                emotion = d_result[0]
                print(f"emotion {emotion}")
                byte_payload = emotion.encode('utf-8')

                send_message(conn, byte_payload)

                #Update the state
                LLM_STATE = LLM_OUTPUT_STATE.CHARACTERS

                #Clear the buffer
                s = d_result[1]

            else: continue

        elif LLM_STATE == LLM_OUTPUT_STATE.CHARACTERS:

            d_result = has_delimiter(s)
            if d_result:
                #Send any final data from the left of the delimiter
                extra_data = d_result[0]
                byte_payload = extra_data.encode('utf-8')
                send_message(conn, byte_payload)

                #Terminate the word streaming state. 
                LLM_STATE = LLM_OUTPUT_STATE.SHOOT
                s2 = '''##TerminateCharacterStreamState##'''
                print(f"sending done from char stream {s2}")
                byte_payload = '''##TerminateCharacterStreamState##'''.encode('utf-8')
                #Send message to JBAZZ that Shoot is next.
                send_message(conn, byte_payload)

                # #Reset delimiter tracker
                # character_stream_delimiter_counter = 0

            #This means that we have a completed word in our buffer. 
            #Delimiter must not be in play otherwise we could send it as text to speak.
            elif re.search(r"""[ ,;:!?.'"]""", s):
                #this means that we need to send this word to JBAZZ
                byte_payload = s.encode('utf-8')
                send_message(conn, byte_payload)
                # print(f"sent s word {s}")
                swords.append(s)
                s = ''

        
        elif LLM_STATE == LLM_OUTPUT_STATE.SHOOT:
            d_result = has_delimiter(s)
            if d_result:
                print(f"Current state of s when SHOOT is ready: {s}")
                #Send the emotion to JBAZZ. We can remove the delimiter. 
                shoot = d_result[0]
                print(f"shoot {shoot}")
                byte_payload = shoot.encode('utf-8')

                send_message(conn, byte_payload)

                #Update the state
                LLM_STATE = LLM_OUTPUT_STATE.NONE

                #Clear the buffer
                s = d_result[1]

                #Break the loop allowing the thread to join
                break
            else: continue

    print(f"thread completed, full LLM response: \n{full_response}\n")
    print(f"Words segmented: {swords}")


def handle_client(conn, addr):
    print(f"Client connected: {addr}")
    character_queue = queue.Queue()
    #Check if ollama is active
    try:
        while True:
            payload = recv_message(conn)
            # print(f"Payload: {payload}")
            #Handle termination of client's connection. 
            if payload is None:
                print("Client disconnected. \nAwaiting new connection.")
                llm.reset()
                break
                

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
                #Run this in a thread b4 the llm ask call. 
                #Iterate over the data queue until llm.ask finishes and the queue it created has been consumed. 
                consume_llm_thread = threading.Thread(target=consume_llm_stream, args=(character_queue, conn))
                consume_llm_thread.start()
                llm_text_output = llm.ask(client_text, character_queue)
                consume_llm_thread.join()

            else:
                print("Error handling client: LLM is not online")
        

            response = f"Processed:  {client_text}@#$ {llm_text_output}"
            send_message(conn, response.encode("utf-8"))
            print(f"Sent message f{response}")

    except ConnectionResetError:
        print("Client crashed / reset connection.")

    finally:
        conn.close()


def run_server():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        s.bind((HOST, PORT))
        s.listen(1)

        print(f"Listening on {HOST}:{PORT}")

        try:
            while True:
                try:
                    conn, addr = s.accept()
                    handle_client(conn, addr)
                except socket.timeout:
                    #This allows us to use KeyboardInterrupt 
                    pass
        except KeyboardInterrupt:
            print("Server closed with KeyboardInterrupt")


if __name__ == "__main__":

    is_sst_online = False
    try:    
        whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
        is_sst_online = True

    except Exception as e:
        print(e) 
    
    is_llm_online = False
    is_llm_online = check_for_llm()
    try:
        llm = LLMContext()
        print(f"LLM is: {is_llm_online}")
    except Exception as e:
        print(e)
        

    run_server()
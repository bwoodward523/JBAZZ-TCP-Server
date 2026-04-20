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


import os
import re
import socket
from llm import * 
from sst import *
from faster_whisper import WhisperModel
import queue
import threading

from tcp_framing import MessageType, recv_message, send_message, send_typed_message


HOST = "0.0.0.0" #Temporary host to listen to all possible connections
PORT = 5555  

# Off by default: stream Kokoro TIMING_DATA + AUDIO_CHUNK after sentence boundaries.
# Set JBAZZ_LLM_WORD_STREAM=1 to restore legacy word-at-a-time UTF-8 frames (no server TTS).
USE_LLM_WORD_STREAM_FALLBACK = os.environ.get("JBAZZ_LLM_WORD_STREAM", "").strip().lower() in ("1", "true", "yes")

server_tts = None

from enum import Enum
#Class to indicate which state of outputting in the LLM we are in. 
class LLM_OUTPUT_STATE(Enum):
    NONE = -1
    EMOTION = 0
    CHARACTERS = 1
    SHOOT = 2
DELIMITER = '@#$'

_SENTENCE_BOUNDARY = re.compile(r"[.!?;\n]")

# Default emotion if the LLM stream ends before the first @#$ (allowed six emotions in SYSTEM_PROMPT).
DEFAULT_EMOTION_EOS = "emotion:surprise"
_LEGACY_TERMINATE = "##TerminateCharacterStreamState##"


def parse_shoot_to_payload(shoot_field: str) -> bytes:
    """Map LLM shoot field ('shoot:True', etc.) to protocol UTF-8 b'True' / b'False'."""
    t = shoot_field.strip()
    if t.startswith("shoot:"):
        t = t[6:].strip()
    low = t.lower()
    if low.startswith("true"):
        return b"True"
    if low.startswith("false"):
        return b"False"
    return b"False"


def _split_response_into_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?;\n])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _emit_eos_phase_gaps(
    conn,
    use_sentence_tts: bool,
    emotion_phase_done: bool,
    dialogue_closed: bool,
    shoot_phase_done: bool,
    llm_state: LLM_OUTPUT_STATE,
    buf: str,
    has_delimiter_fn,
) -> tuple[bool, bool, bool]:
    """Emit missing protocol phases so Pi recv state matches server (truncated LLM stream). Order:
    legacy: EMOTION -> (word frames) -> ##TerminateCharacterStreamState## -> SHOOT field.
    typed: EMOTION -> TIMING_DATA/AUDIO_CHUNK... -> END_OF_RESPONSE -> SHOOT payload.
    Only sends frames that were not yet sent (flags). Returns updated (emotion, dialogue, shoot) flags."""
    patched: list[str] = []

    if not emotion_phase_done:
        if use_sentence_tts:
            send_typed_message(
                conn, MessageType.EMOTION, DEFAULT_EMOTION_EOS.encode("utf-8")
            )
        else:
            send_message(conn, DEFAULT_EMOTION_EOS.encode("utf-8"))
        emotion_phase_done = True
        patched.append("emotion")

    if use_sentence_tts:
        if not dialogue_closed:
            send_typed_message(conn, MessageType.END_OF_RESPONSE, b"")
            dialogue_closed = True
            patched.append("end_of_response")
    else:
        if not dialogue_closed:
            send_message(conn, _LEGACY_TERMINATE.encode("utf-8"))
            dialogue_closed = True
            patched.append("terminate")

    if not shoot_phase_done:
        dr = has_delimiter_fn(buf) if buf else None
        if use_sentence_tts:
            if llm_state == LLM_OUTPUT_STATE.SHOOT and dr is not None:
                send_typed_message(
                    conn, MessageType.SHOOT, parse_shoot_to_payload(dr[0])
                )
            else:
                send_typed_message(conn, MessageType.SHOOT, b"False")
        else:
            if llm_state == LLM_OUTPUT_STATE.SHOOT and dr is not None:
                send_message(conn, dr[0].encode("utf-8"))
            else:
                send_message(conn, b"shoot:False")
        shoot_phase_done = True
        patched.append("shoot")

    if patched:
        print(f"EOS gap fill (synthesized phases): {', '.join(patched)}")

    return emotion_phase_done, dialogue_closed, shoot_phase_done


# Thread for handling the data as it comes
def consume_llm_stream(character_queue, conn, server_tts=None, word_stream_fallback=None):
    """Stream LLM output to the client; mirrors Pi blocking recv state machine.

    Legacy (JBAZZ_LLM_WORD_STREAM): raw emotion UTF-8 -> raw segment frames ->
        ##TerminateCharacterStreamState## -> raw shoot field.
    Kokoro typed: MessageType.EMOTION -> TIMING_DATA + AUDIO_CHUNK per sentence ->
        MessageType.END_OF_RESPONSE -> MessageType.SHOOT (b'True'|'False').
    On EOS, missing phases are filled via _emit_eos_phase_gaps (see plan).
    """

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
        
    if word_stream_fallback is None:
        word_stream_fallback = USE_LLM_WORD_STREAM_FALLBACK
    use_sentence_tts = server_tts is not None and not word_stream_fallback

    def synth_sentence(sentence: str) -> None:
        server_tts.synthesize_and_stream(sentence.strip(), conn, send_typed_message)

    def drain_leading_sentences(buf: str) -> str:
        """Emit complete sentences from the front of buf (streaming path)."""
        while buf:
            if buf.count("@") == 1:
                break
            m = _SENTENCE_BOUNDARY.search(buf)
            if not m:
                break
            sentence = buf[: m.end()].strip()
            buf = buf[m.end() :]
            if sentence:
                synth_sentence(sentence)
        return buf

    if use_sentence_tts:
        server_tts.reset()

    emotion_phase_done = False
    dialogue_closed = False
    shoot_phase_done = False

    LLM_STATE = LLM_OUTPUT_STATE.EMOTION
    s = ''
    full_response = ''
    swords = []
    while True:
        chunk = character_queue.get()
        print(f"{chunk}")

        if chunk is None:
            if LLM_STATE == LLM_OUTPUT_STATE.CHARACTERS:
                rest = (drain_leading_sentences(s) if use_sentence_tts else s).strip()
                if rest:
                    if use_sentence_tts:
                        synth_sentence(rest)
                        send_typed_message(conn, MessageType.END_OF_RESPONSE, b"")
                        dialogue_closed = True
                    else:
                        send_message(conn, rest.encode("utf-8"))
                        swords.append(rest)
                        send_message(conn, _LEGACY_TERMINATE.encode("utf-8"))
                        dialogue_closed = True
                elif use_sentence_tts:
                    send_typed_message(conn, MessageType.END_OF_RESPONSE, b"")
                    dialogue_closed = True
            emotion_phase_done, dialogue_closed, shoot_phase_done = _emit_eos_phase_gaps(
                conn,
                use_sentence_tts,
                emotion_phase_done,
                dialogue_closed,
                shoot_phase_done,
                LLM_STATE,
                s,
                has_delimiter,
            )
            print(f"thread completed (EOS), full LLM response: \n{full_response}\n")
            print(f"Words segmented: {swords}")
            return

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

                if use_sentence_tts:
                    send_typed_message(conn, MessageType.EMOTION, emotion.encode("utf-8"))
                else:
                    send_message(conn, emotion.encode("utf-8"))
                emotion_phase_done = True

                #Update the state
                LLM_STATE = LLM_OUTPUT_STATE.CHARACTERS

                #Clear the buffer
                s = d_result[1]

            else:
                continue

        elif LLM_STATE == LLM_OUTPUT_STATE.CHARACTERS:
            if use_sentence_tts:
                d_result = has_delimiter(s)
                if d_result:
                    print(f"TTS path: shoot delimiter found: {d_result}")
                    extra_data = d_result[0]
                    swords.append(extra_data)
                    for sentence in _split_response_into_sentences(extra_data):
                        synth_sentence(sentence)
                    send_typed_message(conn, MessageType.END_OF_RESPONSE, b"")
                    dialogue_closed = True
                    LLM_STATE = LLM_OUTPUT_STATE.SHOOT
                    s = d_result[1]
                    continue

                s = drain_leading_sentences(s)
                continue

            # Legacy word-at-a-time streaming
            r = re.search(r"""[ ,;:!?."]""", s)
            d_result = has_delimiter(s)
            if d_result:
                print(f"D_result from characters delimiter found: {d_result}")
                extra_data = d_result[0]
                swords.append(extra_data)

                send_message(conn, extra_data.encode('utf-8'))
                s = d_result[1]
                LLM_STATE = LLM_OUTPUT_STATE.SHOOT
                print("sending done from char stream ##TerminateCharacterStreamState##")
                send_message(conn, '''##TerminateCharacterStreamState##'''.encode('utf-8'))
                dialogue_closed = True
                continue

            if s.count('@') == 1:
                continue

            elif r:
                complete = s[:r.end()]
                if complete:
                    send_message(conn, complete.encode('utf-8'))
                    swords.append(complete)
                s = s[r.end():]

        
        elif LLM_STATE == LLM_OUTPUT_STATE.SHOOT:
            d_result = has_delimiter(s)
            if d_result:
                print(f"Current state of s when SHOOT is ready: {s}")
                #Send the emotion to JBAZZ. We can remove the delimiter. 
                shoot = d_result[0]
                print(f"shoot {shoot}")

                if use_sentence_tts:
                    send_typed_message(conn, MessageType.SHOOT, parse_shoot_to_payload(shoot))
                else:
                    send_message(conn, shoot.encode('utf-8'))
                shoot_phase_done = True

                #Update the state
                LLM_STATE = LLM_OUTPUT_STATE.NONE

                #Clear the buffer
                s = d_result[1]

                # Drain any remaining chunks from the producer until the None
                # sentinel, so leftover data does not leak into the next turn's
                # queue (the producer keeps streaming trailing tokens + None
                # after this point).
                while True:
                    trailing = character_queue.get()
                    if trailing is None:
                        break
                break
            else:
                continue

    print(f"thread completed, full LLM response: \n{full_response}\n")
    print(f"Words segmented: {swords}")


def handle_client(conn, addr):
    print(f"Client connected: {addr}")
    character_queue = queue.Queue()
    #Check if ollama is active
    #Clear the chat history
    llm.reset()
    

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
                consume_llm_thread = threading.Thread(
                    target=consume_llm_stream,
                    args=(character_queue, conn, server_tts),
                )
                consume_llm_thread.start()
                llm_text_output = llm.ask(client_text, character_queue)
                consume_llm_thread.join()

            else:
                print("Error handling client: LLM is not online")
        

            # response = f"Processed:  {client_text}@#$ {llm_text_output}"
            # send_message(conn, response.encode("utf-8"))
            # print(f"Sent message f{response}")

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

    if USE_LLM_WORD_STREAM_FALLBACK:
        print("JBAZZ_LLM_WORD_STREAM enabled: legacy word-at-a-time TCP output (no Kokoro).")
    else:
        try:
            from tts import ServerTTS

            server_tts = ServerTTS()
            print("ServerTTS (Kokoro): sentence streaming + typed audio/timing frames.")
        except Exception as e:
            print(f"ServerTTS unavailable ({e}); falling back to legacy word streaming.")
            server_tts = None

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
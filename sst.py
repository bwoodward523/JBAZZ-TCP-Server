import io

def convert_to_text(payload, whisper_model):
    print("Implement convert to text")
    #Convert the bytes into an io buffer
    buffer = io.BytesIO(payload)

    #Name the buffer as a file so whisper accepts it
    buffer.name = "audio.mp3"

    print("Audio received, transcribing...")

    #Sending the buffer to whisper.
    segments, info = whisper_model.transcribe(buffer, language="en")

    #Get the response and turn it into a string
    transcript = " ".join(seg.text.strip() for seg in segments)

    return transcript
    
def test_audio():
    from faster_whisper import WhisperModel

    whisper_model = WhisperModel("base", device="cpu", compute_type="float16")


    with open("test.mp3", "rb") as f:
        audio = f.read()
    
    transcript = convert_to_text(audio, whisper_model)
    print(transcript)


if __name__ == "__main__":
    test_audio()
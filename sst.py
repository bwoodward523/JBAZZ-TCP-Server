import io

def convert_to_text(payload, whisper_model):

   
    print("Implement convert to text")
    #Convert the bytes into an io buffer

    buffer = io.BytesIO(payload)

    #Name the buffer as a file so whisper accepts it
    buffer.name = "audio.mp3"

    print("Audio received, transcribing...")

    assert whisper_model != None
    #Sending the buffer to whisper.
    segments, info = whisper_model.transcribe(buffer, language="en")

    #Get the response and turn it into a string
    transcript = " ".join(seg.text.strip() for seg in segments)

    return transcript
    
def test_audio():
    from faster_whisper import WhisperModel

    whisper_model = WhisperModel("base", device="cuda", compute_type="float16")


    with open("output.wav", "rb") as f:
        audio = f.read()
        print(audio)
    transcript = convert_to_text(audio, whisper_model)
    print(transcript)

def save_debug_wav():
    import wave
    wf = wave.open("output.wav", 'wb')
    wf.setnchannels(self.channels)
    wf.setsampwidth(self.p.get_sample_size(self.sample_format))
    wf.setframerate(self.fs)
    wf.writeframes(bytes)
    wf.close()
if __name__ == "__main__":
    test_audio()
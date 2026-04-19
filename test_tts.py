from kokoro import KPipeline
import numpy as np
import soundfile as sf
import sounddevice as sd
import sys

def main():
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello, this is a test of the Kokoro text to speech pipeline."
    voice = "am_michael"
    speed = 1.0
    output_file = "test_output.wav"
    sample_rate = 24000

    print(f"Initializing KPipeline...")
    pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='a')

    print(f"Synthesizing: \"{text}\"")
    all_audio = []
    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
    stream.start()

    try:
        for i, result in enumerate(pipeline(text, voice=voice, speed=speed)):
            audio = result.audio.cpu().numpy().astype(np.float32)
            all_audio.append(audio)

            duration = len(audio) / sample_rate
            print(f"  Chunk {i}: {len(audio)} samples ({duration:.2f}s)")

            if result.tokens:
                for t in result.tokens:
                    if t and t.start_ts is not None and t.end_ts is not None and t.text:
                        print(f"    [{t.start_ts:.3f} - {t.end_ts:.3f}] {t.text}")

            stream.write(audio)
    finally:
        stream.stop()
        stream.close()

    combined = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)
    sf.write(output_file, combined, sample_rate)
    print(f"\nWrote {len(combined)} samples ({len(combined)/sample_rate:.2f}s) to {output_file}")

if __name__ == "__main__":
    main()

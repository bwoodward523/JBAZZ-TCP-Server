from kokoro import KPipeline
import numpy as np
import struct
import json

class ServerTTS:
    def __init__(self, voice="af_heart", speed=1.0, lang_code="a"):
        self.pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code=lang_code)
        self.voice = voice
        self.speed = speed
        self.audio_duration = 0.0  # cumulative offset across sentences

    def reset(self):
        self.audio_duration = 0.0

    def synthesize_and_stream(self, sentence, conn, send_typed_message):
        """Synthesize a sentence, streaming audio+timings to the Pi."""
        generator = self.pipeline(sentence, voice=self.voice, speed=self.speed)
        
        for result in generator:
            audio_float32 = result.audio.cpu().numpy()
            tokens = result.tokens
            
            # Build timing list with cumulative offsets
            timings = []
            if tokens:
                for t in tokens:
                    if t and t.start_ts is not None and t.end_ts is not None and t.text:
                        timings.append({
                            "s": round(t.start_ts + self.audio_duration, 4),
                            "e": round(t.end_ts + self.audio_duration, 4),
                            "w": t.text
                        })
            
            # Send timing data BEFORE audio so Pi can prepare visemes
            if timings:
                send_typed_message(conn, 0x02, json.dumps(timings).encode('utf-8'))
            
            # Convert and send audio
            audio_int16 = (audio_float32 * 32767).astype(np.int16).tobytes()
            send_typed_message(conn, 0x01, audio_int16)
            
            # Track cumulative duration
            self.audio_duration += len(audio_float32) / 24000
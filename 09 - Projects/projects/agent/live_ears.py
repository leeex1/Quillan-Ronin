import sounddevice as sd, whisper, queue, sys
print("Quillan EARS live: Whisper tiny, listening...")
try:
    model=whisper.load_model("tiny", device="cpu")
    q=queue.Queue()
    def cb(indata, frames, time, status): q.put(indata.copy())
    with sd.InputStream(samplerate=16000, channels=1, callback=cb, blocksize=16000):
        while True:
            audio=q.get(); audio=audio.flatten().astype("float32")
            result=model.transcribe(audio, language="en", fp16=False)
            text=result["text"].strip()
            if text: print(f"HEARD: {text}", flush=True); open("C:/02_QUILLAN/chatlogs/live_heard.txt","a",encoding="utf-8").write(text+"\n")
except Exception as e: print(f"ears error: {e}")

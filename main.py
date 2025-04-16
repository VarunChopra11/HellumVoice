import os
import threading
import time
from pvrecorder import PvRecorder
import pvporcupine
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Environment variables
PV_ACCESS_KEY = os.getenv("PV_ACCESS_KEY")

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://hellumgpt.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-05-01-preview",
)

class CommandThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        
    def stop(self):
        self._stop_event.set()
        
    def stopped(self):
        return self._stop_event.is_set()
    
    def get_gpt_response(self, user_input):
        """Get streaming response from Azure OpenAI GPT"""
        try:
            # Create the system message
            messages = [
                {"role": "system", "content": "You are Hellum, a helpful campus tour assistant. Keep responses concise and informative."},
                {"role": "user", "content": user_input}
            ]
            
            print("\nGetting response from GPT...")
            print("-" * 40)
            
            # Stream the response
            full_response = ""
            completion = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=800,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True
            )
            
            for chunk in completion:
                if self.stopped():
                    print("\nGPT response interrupted.")
                    break
                
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print("\n" + "-" * 40)
            return full_response
        except Exception as e:
            print(f"\nError getting GPT response: {e}")
            return "Sorry, I encountered an error processing your request."
        
    def run(self):
        print("Listening for command...")
        try:
            # Initialize speech recognition
            speech_config = speechsdk.SpeechConfig(
                subscription=AZURE_SPEECH_KEY, 
                region=AZURE_SPEECH_REGION
            )
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
            
            # Set up the event handlers for real-time transcription
            recognized_text = None
            done = False
            
            def stop_cb(evt):
                nonlocal done
                done = True
                
            def recognized_cb(evt):
                # This callback is called for final recognition results
                nonlocal recognized_text
                recognized_text = evt.result.text
                print(f"\nCommand detected: {recognized_text}")
                
            def recognizing_cb(evt):
                # This callback is called for interim results during recognition
                if not self.stopped():
                    print(f"Recognizing: {evt.result.text}", end="\r")
            
            # Connect the event handlers
            speech_recognizer.recognized.connect(recognized_cb)
            speech_recognizer.recognizing.connect(recognizing_cb)
            speech_recognizer.session_stopped.connect(stop_cb)
            speech_recognizer.canceled.connect(stop_cb)
            
            # Start continuous recognition
            speech_recognizer.start_continuous_recognition()
            
            # Wait for recognition to complete or timeout after 5 seconds of silence
            start_time = time.time()
            silent_time = 0
            
            while not done and not self.stopped():
                time.sleep(0.1)
                if recognized_text:
                    break
                
                if time.time() - start_time > 10:  # Total timeout
                    print("\nCommand recognition timed out.")
                    break
            
            # Stop recognition
            speech_recognizer.stop_continuous_recognition()
            
            # Process the recognized text with GPT if available
            if recognized_text and not self.stopped():
                self.get_gpt_response(recognized_text)
                
        except Exception as e:
            print(f"\nCommand recognition error: {e}")

def main():
    try:
        # Initialize Porcupine for wake word detection
        porcupine = pvporcupine.create(
            access_key=PV_ACCESS_KEY,
            keyword_paths=["hellum_win.ppn"],
            sensitivities=[0.7]
        )
        
        # Initialize audio recorder
        recorder = PvRecorder(
            frame_length=porcupine.frame_length,
            device_index=-1  # Default device
        )
        recorder.start()
        
        print("Listening for wake word 'Hellum'... (press Ctrl+C to exit)")
        
        current_command_thread = None
        
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            
            if result >= 0:
                print("\nWake word detected!")
                
                # Pause recording during command processing
                recorder.stop()
                
                # If a command is already being processed, stop it
                if current_command_thread and current_command_thread.is_alive():
                    current_command_thread.stop()
                    current_command_thread.join(timeout=1)
                    print("Previous command interrupted.")
                    
                # Start new command processing thread
                current_command_thread = CommandThread()
                current_command_thread.start()
                
                # Wait for command processing to complete (with timeout for safety)
                current_command_thread.join(timeout=30)
                
                # Resume wake word detection
                if not recorder.is_recording:
                    recorder.start()
                print("\nResumed listening for wake word...")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        if 'current_command_thread' in locals() and current_command_thread and current_command_thread.is_alive():
            current_command_thread.stop()
            current_command_thread.join(timeout=1)
            
        if 'recorder' in locals():
            recorder.delete()
        if 'porcupine' in locals():
            porcupine.delete()
        print("Resources released.")

if __name__ == "__main__":
    main()
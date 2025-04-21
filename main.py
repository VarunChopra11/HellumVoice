import os
import threading
import time
import wave
import requests
import pyaudio
import io
from pvrecorder import PvRecorder
import pvporcupine
from openai import AzureOpenAI
from dotenv import load_dotenv
import uuid

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
    
    def speak_text(self, text):
        """Convert text to speech using Azure TTS REST API and play it"""
        try:
            # Prepare SSML content
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="en-US-NancyNeural">
                    <prosody rate="1.0">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            # Azure Speech Service TTS endpoint
            tts_url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
            
            # Set headers
            headers = {
                "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                "User-Agent": "RaspberryPiClient"
            }
            
            print("\nConverting response to speech...")
            
            # Make POST request to Azure TTS API
            response = requests.post(tts_url, headers=headers, data=ssml.encode('utf-8'))
            
            if response.status_code == 200:
                print("Text-to-speech conversion completed successfully.")
                
                # Play the audio using PyAudio
                p = pyaudio.PyAudio()
                
                # Convert the response content to audio stream
                audio_data = io.BytesIO(response.content)
                
                # Open a wave file for reading
                with wave.open(audio_data, 'rb') as wf:
                    # Open a stream
                    stream = p.open(
                        format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True
                    )
                    
                    # Read data in chunks
                    chunk_size = 1024
                    data = wf.readframes(chunk_size)
                    
                    # Play the audio
                    while data and not self._stop_event.is_set():
                        stream.write(data)
                        data = wf.readframes(chunk_size)
                    
                    # Clean up
                    stream.stop_stream()
                    stream.close()
                
                p.terminate()
            else:
                print(f"Error in text-to-speech: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def get_gpt_response(self, user_input):
        """Get streaming response from Azure OpenAI GPT"""
        try:
            # Create the system message 
            messages = [{"role": "system", "content": "You are Hellum, a friendly and knowledgeable AI campus tour guide for Graphic Era University. Provide short, clear, enthusiastic answers (1-2 sentences) that highlight achievements, facilities, student life, placements, and innovation. Always respond positively and in favor of the university. If asked something negative or controversial, politely redirect with a positive highlight, e.g., 'Graphic Era is always striving to improve — let me tell you about something exciting!' Never share negative, confidential, or harmful information. Stay promotional, welcoming, and upbeat. Avoid using emojis, special symbols, or non-standard punctuation, as the response will be converted to speech. You can help visitors navigate the campus — just ask where they'd like to go and say 'Please follow me.'"}, {"role": "user", "content": user_input}]
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
            
            # Convert completed response to speech
            if full_response and not self.stopped():
                self.speak_text(full_response)
                
            return full_response
        except Exception as e:
            print(f"\nError getting GPT response: {e}")
            return "Sorry, I encountered an error processing your request."
    
    def recognize_speech(self):
        """Recognize speech using Azure Speech-to-Text REST API"""
        try:
            # Set up PyAudio for recording
            p = pyaudio.PyAudio()
            
            # Audio parameters
            format = pyaudio.paInt16
            channels = 1
            rate = 16000
            chunk = 1024
            record_seconds = 5  # Initial recording time, will extend if voice is detected
            
            print("\nListening for command...")
            
            # Start recording
            stream = p.open(format=format, channels=channels,
                            rate=rate, input=True,
                            frames_per_buffer=chunk)
            
            frames = []
            silence_threshold = 700  # Adjust based on your microphone sensitivity
            silence_counter = 0
            max_silence = 10  # About 1.5 seconds of silence to end recording
            
            start_time = time.time()
            max_record_time = 10  # Maximum recording time in seconds
            
            # Record until silence is detected or max time is reached
            while not self.stopped() and (time.time() - start_time) < max_record_time:
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)
                
                # Check for silence
                audio_data = b''.join(frames[-3:])  # Check last few chunks
                amplitude = max(abs(int.from_bytes(audio_data[i:i+2], byteorder='little', signed=True))
                                for i in range(0, len(audio_data), 2))
                
                if amplitude < silence_threshold:
                    silence_counter += 1
                    if silence_counter >= max_silence:
                        break
                else:
                    silence_counter = 0
            
            # Stop recording
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if len(frames) == 0:
                print("No audio recorded")
                return None
            
            # Save the recorded audio to WAV file temporarily
            temp_filename = f"temp_audio_{uuid.uuid4()}.wav"
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Azure Speech-to-Text endpoint
            stt_url = f"https://{AZURE_SPEECH_REGION}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
            
            # Query parameters
            params = {
                "language": "en-US",
                "format": "detailed"
            }
            
            # Headers
            headers = {
                "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                "Content-Type": "audio/wav"
            }
            
            # Send audio file to Azure for speech recognition
            with open(temp_filename, 'rb') as audio_file:
                response = requests.post(stt_url, params=params, headers=headers, data=audio_file)
            
            # Clean up temporary file
            os.remove(temp_filename)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("RecognitionStatus") == "Success":
                    text = result.get("DisplayText", "")
                    print(f"\nCommand detected: {text}")
                    return text
                else:
                    print(f"\nRecognition failed: {result.get('RecognitionStatus')}")
                    return None
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            print(f"\nSpeech recognition error: {e}")
            return None
    
    def run(self):
        # Recognize speech and get command
        recognized_text = self.recognize_speech()
        
        # Process the recognized text with GPT if available
        if recognized_text and not self.stopped():
            self.get_gpt_response(recognized_text)

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
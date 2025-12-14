import gradio as gr
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
from fer import FER
import whisper
from textblob import TextBlob
import os
import pathlib
import requests
import tempfile
import shutil
import subprocess
import sys
import uuid

# Check FFmpeg availability
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        raise EnvironmentError("FFmpeg is not installed or not in PATH. Please install FFmpeg.")

check_ffmpeg()

# Ensure FER model is present
FER_MODEL_PATH = (
    pathlib.Path(__import__('fer').__path__[0]) / "data" / "emotion_model.hdf5"
)
if not FER_MODEL_PATH.exists():
    url = "https://github.com/justinshenk/fer/releases/download/v-0.1/emotion_model.hdf5"
    FER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url)
    FER_MODEL_PATH.write_bytes(r.content)

HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
whisper_model = whisper.load_model("base")
emotion_detector = FER(mtcnn=False)

def process_video(input_video):
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_video = os.path.join(tmpdir, "temp_video.mp4")
        processed_video = os.path.join(tmpdir, "processed_video.mp4")
        extracted_audio = os.path.join(tmpdir, "extracted_audio.wav")
        final_video = os.path.join(tmpdir, "final_processed_video.mp4")

        shutil.copy(input_video, temp_video)

        cap = cv2.VideoCapture(temp_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(processed_video, fourcc, fps, (w, h))
        face_cascade = cv2.CascadeClassifier(HAAR_PATH)

        SKIP_FRAMES = 2
        frames_written = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % SKIP_FRAMES == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) == 0:
                    writer.write(frame)
                    frames_written += 1
                    frame_idx += 1
                    continue
                for (x, y, fw, fh) in faces:
                    face_img = frame[y : y + fh, x : x + fw]
                    label, score = emotion_detector.top_emotion(face_img)
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                    if label:
                        cv2.putText(
                            frame,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
            writer.write(frame)
            frames_written += 1
            frame_idx += 1
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        if not os.path.exists(processed_video) or os.path.getsize(processed_video) == 0 or frames_written == 0:
            return None, "Processed video was not created.", ""

        video_clip = VideoFileClip(temp_video)
        if video_clip.audio is not None:
            video_clip.audio.write_audiofile(extracted_audio)
        else:
            return None, "No audio track found in the uploaded video.", ""
        video_clip.close()

        if not os.path.exists(extracted_audio) or os.path.getsize(extracted_audio) == 0:
            return None, "Audio extraction failed.", ""

        transcription = whisper_model.transcribe(extracted_audio)["text"]
        polarity = TextBlob(transcription).sentiment.polarity
        overall = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        sentiment_result = f"Overall Sentiment: {overall}"

        processed_clip = VideoFileClip(processed_video)
        audio_clip = AudioFileClip(extracted_audio)
        final_clip = processed_clip.set_audio(audio_clip)
        final_clip.write_videofile(final_video, codec="libx264", preset="ultrafast")
        processed_clip.close()
        audio_clip.close()
        final_clip.close()

        # Save the final video to a persistent location with a unique name
        persistent_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(persistent_dir, exist_ok=True)
        unique_name = f"processed_{uuid.uuid4().hex}.mp4"
        persistent_final_video = os.path.join(persistent_dir, unique_name)
        shutil.copy(final_video, persistent_final_video)

        return persistent_final_video, sentiment_result, transcription

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video (MP4)"),
    outputs=[
        gr.Video(label="Processed Video with Emotions"),
        gr.Textbox(label="Overall Sentiment"),
        gr.Textbox(label="Transcription"),
    ],
    title="Emotion and Sentiment Detection from Video",
    description="Upload a video file. The app will detect emotions in faces, transcribe the audio, and analyze sentiment."
)

if __name__ == "__main__":
    iface.launch()
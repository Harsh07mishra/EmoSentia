import streamlit as st
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
from fer import FER  # Emotion detection
import whisper
from textblob import TextBlob
import os
import re

# Initialize Whisper model for speech-to-text transcription
whisper_model = whisper.load_model("base")

# Streamlit app layout
st.title("Emotion and Sentiment Detection from Video")
st.sidebar.header("Upload Your Video")

# File upload functionality
uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Video uploaded successfully!")

    # Load video file
    video_clip = VideoFileClip("temp_video.mp4")
    audio_clip = video_clip.audio  # Keep the audio from the original video

    # Initialize OpenCV video capture
    cap = cv2.VideoCapture("temp_video.mp4")
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get the frame rate
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frame count
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frames
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frames

    # Prepare video writer to save processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    processed_video = cv2.VideoWriter('processed_video.mp4', fourcc, frame_rate, (width, height))

    # Initialize FER emotion detector
    emotion_detector = FER()

    # Process each frame
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces and emotions in the frame using OpenCV Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Get the emotion
            emotions, _ = emotion_detector.top_emotion(frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
            cv2.putText(frame, emotions, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the processed frame to the output video
        processed_video.write(frame)

    # Release the OpenCV video capture and writer
    cap.release()
    processed_video.release()

    # Extract audio and save to file
    audio_file = "extracted_audio.wav"
    video_clip.audio.write_audiofile(audio_file)

    # Speech-to-text transcription using Whisper
    transcription = whisper_model.transcribe(audio_file)
    transcribed_text = transcription["text"]
    
    # Sentiment analysis on the transcribed text
    blob = TextBlob(transcribed_text)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        overall_sentiment = "Positive"
    elif sentiment < 0:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    # Display the overall sentiment
    st.subheader(f"Overall Sentiment: {overall_sentiment}")

    # Option to display contributing sentences for sentiment
    show_sentences = st.checkbox("Show Sentences Contributing to Sentiment")

    if show_sentences:
        st.subheader("Contributing Sentences and Their Sentiments:")
        # Split the transcribed text into sentences
        sentences = re.split(r'\. |\? |\! ', transcribed_text)
        for sentence in sentences:
            sentence_blob = TextBlob(sentence)
            sentence_sentiment = sentence_blob.sentiment.polarity
            if sentence_sentiment > 0:
                sentiment_label = "Positive"
            elif sentence_sentiment < 0:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

            st.write(f"Sentence: {sentence}")
            st.write(f"Sentiment: {sentiment_label}")
            st.write("---")

    # Reattach the original audio to the processed video
    final_video = VideoFileClip("processed_video.mp4")
    final_audio = AudioFileClip("temp_video.mp4")
    final_video = final_video.set_audio(final_audio)

    # Write the final output to a file
    final_video.write_videofile("final_processed_video.mp4", codec="libx264")

    # Show the processed video to the user
    st.subheader("Processed Video with Emotion Detection")
    st.video("final_processed_video.mp4")

    # Option to download the processed video
    with open("final_processed_video.mp4", "rb") as f:
        st.download_button("Download Processed Video", f, file_name="processed_video.mp4")

    # Clean up temporary files
    os.remove("temp_video.mp4")
    os.remove("processed_video.mp4")
    os.remove("final_processed_video.mp4")
    os.remove(audio_file)

import streamlit as st
import cv2
import numpy as np
from fer import FER
import matplotlib.pyplot as plt

def main():
    st.title("ExpressionAI")
    st.header('Facial Expression Feedback Trainer')
    st.write('Bollywood dance places a high emphasis on facial expressions, and if you want to improve your facial expressions while performing then you are in the right place! Upload a dance video (<1 minute for best results) with a clear view of your face, and get ready to have your mind blown!')
    # Upload the video file
    video_file = st.file_uploader("Upload video", type=["mp4"])

    if video_file is not None:
        # Create a temporary file to save the uploaded video
        temp_video_path = "temp.mp4"
        with open(temp_video_path, "wb") as temp_video_file:
            temp_video_file.write(video_file.read())

        # Read the video file
        video = cv2.VideoCapture(temp_video_path)
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate) * 10  # Process every 10 seconds

        # Create FER model
        detector = FER()

        # Process video frames
        st.text("Processing video...")
        frames = []
        timestamps = []
        frame_count = 0  # Counter variable
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 != 0:
                continue  # Skip frames that are not multiples of 10

            # Process the frame and detect facial expressions
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_emotions(processed_frame)

            # Extract timestamp
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 500.0
            timestamps.append(timestamp)

            # Extract dominant emotion
            if result and len(result) > 0:
                emotion = max(result[0]["emotions"], key=result[0]["emotions"].get)
                frames.append(emotion)
            else:
                frames.append("neutral")
            # Draw rectangle and display the processed frame
            if result and len(result) > 0:
                bounding_box = result[0]["box"]
                x, y, w, h = bounding_box
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    processed_frame,
                    emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            with st.sidebar:
                st.image(processed_frame, channels="RGB")

        video.release()


        #Column 1 & Column 2 for Graphs
        col1, col2 = st.columns(2)
        user_colour = st.color_picker(label='Choose a colour for your plot')
        with col1:
            # Generate graph
            emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            filtered_timestamps = timestamps[::10]  # Take timestamps at intervals of 10 frames
            filtered_frames = [frames[i] for i in range(len(frames)) if i % 10 == 0]  # Filter frames at intervals of 10 frames

            # Count the occurrences of each emotion
            emotion_counts = {emotion: filtered_frames.count(emotion) for emotion in emotion_labels}

            fig, ax = plt.subplots()
            ax.bar(emotion_labels, [emotion_counts[emotion] for emotion in emotion_labels], color=user_colour)
            ax.set(xlabel="Emotion", ylabel="Count", title="Emotion Counts")
            ax.grid(True)
            
            # Display the graph
            st.pyplot(fig)
            

        with col2:
            if len(timestamps) > 0:
            
                # fig, ax = plt.subplots()
              
                # ax.plot(timestamps, frames, color=user_colour)
                # ax.set(xlabel="Time (s)", ylabel="Emotion", title="Emotion over time")

                # ax.set_xticks(np.arange(0, timestamps[-1], 10))
                # ax.set_yticks(range(len(frames)))
                # ax.set_yticklabels(list(set(frames)), rotation="vertical", fontsize='small', ha="center")

                # ax.yaxis.labelpad=20
                # ax.xaxis.labelpad=20
                # ax.set_ylim(0,7)
                # ax.grid(True)

                fig, ax = plt.subplots()
                ax.plot(timestamps, frames)
                ax.set(xlabel="Time (s)", ylabel="Emotion", title="Emotion over time")
                ax.set_xticks(np.arange(0, timestamps[-1], 10))
                unique_emotions = list(set(frames))
                ax.set_yticks(range(len(unique_emotions)))
                ax.set_yticklabels(unique_emotions)
                ax.tick_params(axis='y', pad=10)  # Adjust the padding between ticks and tick labels
                ax.grid(True)
                
                # Display the graph
                st.pyplot(fig, figsize=(5,5))
            else:
                # Display a message if there are no timestamps
                st.text("No frames to display.")

if __name__ == "__main__":
    main()
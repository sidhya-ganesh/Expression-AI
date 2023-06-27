# import streamlit as st
# import cv2
# import numpy as np
# from fer import FER
# import matplotlib.pyplot as plt

# def main():
#     st.title("Facial Expression Recognition")

#     # Upload the video file
#     video_file = st.file_uploader("Upload video", type=["mp4"])

#     if video_file is not None:
#         # Create a temporary file to save the uploaded video
#         temp_video_path = "temp.mp4"
#         with open(temp_video_path, "wb") as temp_video_file:
#             temp_video_file.write(video_file.read())

#         # Read the video file
#         video = cv2.VideoCapture(temp_video_path)
#         frame_rate = video.get(cv2.CAP_PROP_FPS)
#         frame_interval = int(frame_rate) * 10  # Process every 10 seconds

#         # Create FER model
#         detector = FER()

#         # Process video frames
#         st.text("Processing video...")
#         frames = []
#         timestamps = []
#         frame_count = 0  # Counter variable
#         while True:
#             ret, frame = video.read()
#             if not ret:
#                 break

#             frame_count += 1
#             if frame_count % 10 != 0:
#                 continue  # Skip frames that are not multiples of 10

#             # Process the frame and detect facial expressions
#             processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = detector.detect_emotions(processed_frame)

#             # Extract timestamp
#             timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 10000.0
#             timestamps.append(timestamp)

#             # Extract dominant emotion
#             if result and len(result) > 0:
#                 emotion = max(result[0]["emotions"], key=result[0]["emotions"].get)
#                 frames.append(emotion)
#             else:
#                 frames.append(None)

#             # Draw rectangle and display the processed frame
#             if result and len(result) > 0:
#                 bounding_box = result[0]["box"]
#                 x, y, w, h = bounding_box
#                 cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(
#                     processed_frame,
#                     emotion,
#                     (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9,
#                     (0, 255, 0),
#                     2,
#                     cv2.LINE_AA,
#                 )

#             st.image(processed_frame, channels="RGB")

#         video.release()

#         # Generate graph
#         emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
#         filtered_timestamps = timestamps[::10]  # Take timestamps at intervals of 10 frames
#         filtered_frames = [frames[i] for i in range(len(frames)) if i % 10 == 0]  # Filter frames at intervals of 10 frames

#         # Count the occurrences of each emotion
#         emotion_counts = {emotion: filtered_frames.count(emotion) for emotion in emotion_labels}

#         fig, ax = plt.subplots()
#         ax.bar(emotion_labels, [emotion_counts[emotion] for emotion in emotion_labels])
#         ax.set(xlabel="Emotion", ylabel="Count", title="Emotion Counts")
#         ax.grid(True)

#         # Display the graph
#         st.pyplot(fig)

# if __name__ == "main":
#     main()

import streamlit as st
import cv2
import numpy as np
from fer import FER
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.title("Facial Expression Recognition")

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
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 10000.0
            timestamps.append(timestamp)

            # Extract dominant emotion
            if result and len(result) > 0:
                emotion = max(result[0]["emotions"], key=result[0]["emotions"].get)
                frames.append(emotion)
            else:
                frames.append(None)

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

            st.image(processed_frame, channels="RGB")

        video.release()

        # Generate graph
        emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        filtered_timestamps = timestamps[::10]  # Take timestamps at intervals of 10 frames
        filtered_frames = [frames[i] for i in range(len(frames)) if i % 10 == 0]  # Filter frames at intervals of 10 frames

        # Count the occurrences of each emotion
        emotion_counts = {emotion: filtered_frames.count(emotion) for emotion in emotion_labels}

        fig, ax = plt.subplots()
        ax.bar(emotion_labels, [emotion_counts[emotion] for emotion in emotion_labels])
        ax.set(xlabel="Emotion", ylabel="Count", title="Emotion Counts")
        ax.grid(True)

        # Display the graph
        st.pyplot(fig)

        #Line Chart
        # chart_data = pd.DataFrame(
        # emotion_counts['happy'], columns=(emotion_labels))        
        # st.line_chart(chart_data)
        # st.write(emotion_counts)

        #code 2
        # fig, ax = plt.subplots()
        # ax.plot(timestamps, frames, marker='o')
        # ax.set(xlabel="Time (s)", ylabel="Emotion", title="Emotion over time")
        # ax.set_xticks(np.arange(0, timestamps[-1], 10))
        # ax.set_yticks(range(len(detector.emotion_labels)))
        # ax.set_yticklabels(detector.emotion_labels)
        # ax.grid(True)

        # Display the graph
        # st.pyplot(fig)

        # st.write(emotion_counts)
        # chart_data = pd.DataFrame(
        # [[emotion_counts['angry']], [emotion_counts['disgust']], [emotion_counts['fear']], [emotion_counts['happy']], [emotion_counts['sad']], [emotion_counts['surprise']], [emotion_counts['neutral']]],
        # columns=emotion_labels)

        # st.line_chart(chart_data)

         # Generate graph
        # fig, ax = plt.subplots()
        # ax.plot(timestamps, frames, marker='o', linestyle='-', linewidth=2)
        # ax.set(xlabel="Time (s)", ylabel="Emotion", title="Emotion over time")
        # ax.set_xticks(np.arange(0, timestamps[-1], 10))
        # ax.set_yticks(range(len(detector.emotion_labels)))
        # ax.set_yticklabels(detector.emotion_labels)
        # ax.grid(True)

        # # Display the graph
        # st.pyplot(fig)

        if len(timestamps) > 0:
            # Generate graph if there are timestamps
            fig, ax = plt.subplots()
            ax.plot(timestamps, frames, marker='o', linestyle='-', linewidth=2)
            ax.set(xlabel="Time (s)", ylabel="Emotion", title="Emotion over time")
            ax.set_xticks(np.arange(0, timestamps[-1], 10))
            ax.set_yticks(range(len(detector.emotion_labels)))
            ax.set_yticklabels(detector.emotion_labels)
            ax.grid(True)

            # Display the graph
            st.pyplot(fig)
        else:
            # Display a message if there are no timestamps
            st.text("No frames to display.")

if __name__ == "__main__":
    main()
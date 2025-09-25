import streamlit as st
import os
import tempfile
import moviepy.editor as mp
import cv2
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

# =============================
# 1. Page config
# =============================
st.set_page_config(
    page_title="🎬 Video Q&A with LLM",
    page_icon="🎥",
    layout="wide"
)

st.title("🎬 Video → Audio + Frames → Q&A with LLM")
st.markdown("**Upload a video, extract text & frames, and ask questions!** 🚀")

# =============================
# 2. File uploader
# =============================
uploaded_video = st.file_uploader("📤 Upload your video", type=["mp4", "avi", "mov"])

if uploaded_video:
    st.video(uploaded_video)

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_video.name)

    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # =============================
    # 3. Extract Audio
    # =============================
    if st.button("🎙 Extract Audio"):
        audio_path = os.path.join(temp_dir, "audio.wav")
        video_clip = mp.VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        st.success("✅ Audio extracted successfully!")
        st.audio(audio_path)

        # Transcribe
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
        transcript = asr(audio_path)["text"]
        st.session_state["transcript"] = transcript
        st.text_area("📝 Transcript", transcript, height=150)

    # =============================
    # 4. Extract Frames
    # =============================
    if st.button("🖼 Extract Frames"):
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_count, saved_frames = 0, []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:
                frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)
            frame_count += 1

        cap.release()
        st.success(f"✅ Extracted {len(saved_frames)} frames")
        st.image(saved_frames[:5], caption=[os.path.basename(f) for f in saved_frames[:5]], width=200)

        st.session_state["frames"] = saved_frames

    # =============================
    # 5. Image Captioning
    # =============================
    if st.button("📝 Generate Captions"):
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        captions = []

        if "frames" in st.session_state:
            for f in st.session_state["frames"][:5]:
                result = captioner(f)[0]["generated_text"]
                captions.append(result)

            st.session_state["captions"] = captions
            st.success("✅ Captions generated!")
            for c in captions:
                st.write("• " + c)

    # =============================
    # 6. Query Retrieval with LLM
    # =============================
    if st.button("🔍 Build Vector DB"):
        text_data = []

        if "transcript" in st.session_state:
            text_data.append(st.session_state["transcript"])
        if "captions" in st.session_state:
            text_data.extend(st.session_state["captions"])

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(text_data)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))

        st.session_state["vector_db"] = (index, embeddings, text_data)
        st.success("✅ Vector DB built and ready!")

    # =============================
    # 7. Ask Question
    # =============================
    if "vector_db" in st.session_state:
        query = st.text_input("💬 Ask a question about the video")
        if st.button("🤖 Get Answer"):
            index, embeddings, text_data = st.session_state["vector_db"]
            model = SentenceTransformer("all-MiniLM-L6-v2")

            q_emb = model.encode([query])
            D, I = index.search(np.array(q_emb), k=3)

            retrieved = [text_data[i] for i in I[0]]
            st.write("🔎 Retrieved Context:", retrieved)

            # LLM (Ollama Qwen)
            llm = ChatOllama(model="qwen3:0.6b")
            prompt = f"Context: {retrieved}\n\nQuestion: {query}\nAnswer:"
            response = llm.invoke([HumanMessage(content=prompt)])

            st.success("🤖 LLM Answer:")
            st.write(response.content)



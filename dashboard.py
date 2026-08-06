import os
import json
import asyncio
import subprocess

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

import yt_dlp
import edge_tts
import streamlit as st
from faster_whisper import WhisperModel
from google import genai

# Page Config
st.set_page_config(page_title="YouTube Repurposing Engine", page_icon="🎬", layout="wide")

# Load environment variables from a local .env file if available.
# This keeps secrets out of the repo while still allowing automatic loading.
base_dir = os.path.dirname(os.path.abspath(__file__))
if load_dotenv is not None:
    load_dotenv(dotenv_path=os.path.join(base_dir, ".env"), override=False)

# Get API key safely from environment variables.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY environment variable not found. Please set it in your environment or in a local .env file before running the app.")
    st.stop()

ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Available Free Edge-TTS Voices
VOICE_OPTIONS = {
    "Male - Christopher (US)": "en-US-ChristopherNeural",
    "Male - Guy (US)": "en-US-GuyNeural",
    "Female - Ava (US)": "en-US-AvaNeural",
    "Female - Emma (US)": "en-US-EmmaNeural",
    "Male - Ryan (UK)": "en-GB-RyanNeural",
    "Female - Sonia (UK)": "en-GB-SoniaNeural"
}

def download_media(youtube_url: str):
    extractor_options = {'youtube': {'player_client': ['ios', 'android', 'web']}}

    ydl_video_opts = {
        'format': 'bestvideo[height>=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',
        'outtmpl': 'full_video.mp4',
        'extractor_args': extractor_options,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_video_opts) as ydl:
        ydl.download([youtube_url])

    ydl_audio_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': 'downloaded_audio',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'extractor_args': extractor_options,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_audio_opts) as ydl:
        ydl.download([youtube_url])
    
    return "full_video.mp4", "downloaded_audio.mp3"

def transcribe_audio_with_words(audio_file_path: str, model_instance):
    segments, _ = model_instance.transcribe(audio_file_path, beam_size=5, word_timestamps=True)
    formatted_transcript = []
    all_words = []
    
    for segment in segments:
        start_min = int(segment.start // 60)
        start_sec = int(segment.start % 60)
        formatted_transcript.append(f"[{start_min:02d}:{start_sec:02d}] {segment.text}")
        if segment.words:
            for w in segment.words:
                all_words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
    
    return "\n".join(formatted_transcript), all_words

def parse_gemini_json_response(raw_text: str) -> dict:
    cleaned = (raw_text or "").strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned invalid JSON. Response preview: {cleaned[:500]}") from exc


def analyze_transcript_with_gemini(transcript: str) -> dict:
    prompt = f"""
    You are an expert YouTube producer. Analyze the transcript with timestamps.
    Identify the TOP 3 most engaging segments (20 to 45 seconds each) for YouTube Shorts / Reels.
    Extract the EXACT spoken text so we can re-generate it with an AI voiceover.

    IMPORTANT:
    - Output ONLY valid JSON.
    - Do not wrap it in markdown fences.
    - Do not include explanations or notes.
    - Return raw JSON only.

    Return your response strictly in valid JSON format:
    {{
      "highlights": [
        {{
          "start_time": "00:42",
          "end_time": "01:23",
          "hook": "string",
          "script_text": "The exact full text spoken during this segment...",
          "twitter_post": "string"
        }}
      ]
    }}

    Transcript:
    {transcript}
    """

    response = ai_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    return parse_gemini_json_response(response.text)

async def generate_ai_voiceover(text: str, voice_code: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice_code)
    await communicate.save(output_path)

def timestamp_to_seconds(ts_str: str) -> float:
    parts = ts_str.strip().split(":")
    return float(parts[0]) * 60 + float(parts[1])

def format_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def generate_srt_from_ai_words(ai_words: list, srt_path: str):
    srt_lines = []
    chunk_size = 3
    index = 1
    
    for i in range(0, len(ai_words), chunk_size):
        chunk = ai_words[i:i + chunk_size]
        if not chunk: continue
        text = " ".join([w["word"] for w in chunk])
        srt_lines.append(f"{index}")
        srt_lines.append(f"{format_srt_time(chunk[0]['start'])} --> {format_srt_time(chunk[-1]['end'])}")
        srt_lines.append(f"{text}\n")
        index += 1

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

def render_individual_clip(video_file: str, item: dict, clip_index: int, voice_code: str, whisper_model) -> str:
    start_sec = timestamp_to_seconds(item["start_time"])
    end_sec = timestamp_to_seconds(item["end_time"])
    video_duration = end_sec - start_sec

    ai_audio_path = os.path.abspath(f"output_clips/ai_voice_{clip_index}.mp3")
    srt_file = os.path.abspath(f"output_clips/temp_{clip_index}.srt")
    output_clip = os.path.abspath(f"output_clips/clip_{clip_index}.mp4")

    # 1. Generate Voiceover
    asyncio.run(generate_ai_voiceover(item["script_text"], voice_code, ai_audio_path))
    
    # 2. Transcribe AI Audio
    _, ai_words = transcribe_audio_with_words(ai_audio_path, whisper_model)
    generate_srt_from_ai_words(ai_words, srt_file)

    # 3. Windows-safe path escaping for FFmpeg subtitles filter
    escaped_srt = srt_file.replace("\\", "/").replace(":", "\\:")

    filter_complex = (
        f"[0:v]crop=ih*(9/16):ih[v_cropped];"
        f"[v_cropped]subtitles=filename='{escaped_srt}':force_style='Alignment=2,FontSize=20,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2'[v_out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec), 
        "-i", os.path.abspath(video_file),
        "-i", ai_audio_path,
        "-t", str(video_duration),
        "-filter_complex", filter_complex,
        "-map", "[v_out]", 
        "-map", "1:a:0",
        "-c:v", "libx264", 
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest", 
        output_clip
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0 or not os.path.exists(output_clip) or os.path.getsize(output_clip) == 0:
        print(f"❌ Clip {clip_index} rendering failed!")
        if result.stderr.strip():
            print(result.stderr)
        for path in [output_clip, ai_audio_path, srt_file]:
            if os.path.exists(path):
                try: os.remove(path)
                except OSError: pass
        return ""

    # Cleanup temporary files
    for path in [ai_audio_path, srt_file]:
        if os.path.exists(path): 
            try: os.remove(path)
            except OSError: pass

    return output_clip

def concatenate_clips(clip_paths: list, output_final_path: str) -> bool:
    """Stitches multiple valid video clips using filter_complex concat for maximum stability."""
    valid_paths = [
        path for path in clip_paths
        if path and os.path.exists(path) and os.path.getsize(path) > 0
    ]

    if not valid_paths:
        print("❌ No valid clips were found for final concatenation.")
        return False

    if len(valid_paths) != len(clip_paths):
        print(f"⚠️ Skipping {len(clip_paths) - len(valid_paths)} invalid or empty clip(s).")

    inputs = []
    filter_inputs = ""
    
    for idx, path in enumerate(valid_paths):
        inputs.extend(["-i", os.path.abspath(path)])
        filter_inputs += f"[{idx}:v][{idx}:a]"

    num_clips = len(valid_paths)
    filter_complex = f"{filter_inputs}concat=n={num_clips}:v=1:a=1[outv][outa]"

    cmd = [
        "ffmpeg", "-y"
    ] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_final_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("❌ Final compilation failed!")
        if result.stderr.strip():
            print(result.stderr)
        return False

    final_exists = os.path.exists(output_final_path) and os.path.getsize(output_final_path) > 0
    if not final_exists:
        print(f"❌ Final video was not created at {output_final_path}")
    return final_exists

# ================= STREAMLIT DASHBOARD UI =================
st.title("🎬 AI YouTube Repurposing Engine")
st.caption("Convert long-form YouTube videos into vertical AI-voiced compilation shorts for $0.")

# User Inputs
youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
selected_voice_label = st.selectbox("Select AI Voice:", list(VOICE_OPTIONS.keys()))
voice_code = VOICE_OPTIONS[selected_voice_label]

if st.button("🚀 Generate Repurposed Short"):
    if not youtube_url:
        st.warning("Please enter a YouTube video URL.")
    else:
        status = st.status("Running Automation Pipeline...", expanded=True)
        
        # Step 1: Download
        status.write("📥 Downloading YouTube video & audio...")
        video_path, audio_path = download_media(youtube_url)
        
        # Step 2: Transcribe
        status.write("🎙️ Transcribing audio with Whisper...")
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        transcript, _ = transcribe_audio_with_words(audio_path, whisper_model)
        
        # Step 3: Analyze
        status.write("🧠 Extracting viral segments with Gemini AI...")
        insights = analyze_transcript_with_gemini(transcript)
        
        os.makedirs("output_clips", exist_ok=True)
        rendered_clips = []
        
        # Step 4: Render Individual Clips
        highlights = insights.get("highlights", [])
        for idx, item in enumerate(highlights, 1):
            status.write(f"✂️ Rendering Clip {idx}/{len(highlights)} with AI Voice...")
            clip_file = render_individual_clip(video_path, item, idx, voice_code, whisper_model)
            if clip_file:
                rendered_clips.append(clip_file)

        if not rendered_clips:
            status.update(label="⚠️ No valid clips were rendered", state="error", expanded=False)
            st.error("No valid video clips were generated. The individual ffmpeg renders failed.")
            st.stop()
            
        # Step 5: Stitch into 1 Final Video
        status.write("🎞️ Concatenating clips into 1 final output video...")
        final_output_path = "output_clips/FINAL_COMPILATION_SHORT.mp4"
        final_ready = concatenate_clips(rendered_clips, final_output_path)
        
        # Cleanup source downloads
        if os.path.exists(audio_path): os.remove(audio_path)
        if os.path.exists(video_path): os.remove(video_path)

        if not final_ready:
            status.update(label="⚠️ Final compilation failed", state="error", expanded=False)
            st.error("⚠️ Final compilation video could not be generated. Please check terminal logs.")
            st.stop()
        
        status.update(label="✅ Processing Complete!", state="complete", expanded=False)
        
        # Display Results in UI
        st.divider()
        st.subheader("🎥 Final Stitched Compilation Short")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
                with open(final_output_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                st.video(video_bytes)
                
                st.download_button(
                    label="⬇️ Download Final Video",
                    data=video_bytes,
                    file_name="repurposed_compilation.mp4",
                    mime="video/mp4"
                )
            else:
                st.error("⚠️ Final compilation video could not be generated.")
                
        with col2:
            st.subheader("📝 Social Media Post Drafts")
            for idx, item in enumerate(highlights, 1):
                with st.expander(f"Clip {idx}: {item['hook']}"):
                    st.write(f"**Script:** {item['script_text']}")
                    st.code(item['twitter_post'], language="markdown")
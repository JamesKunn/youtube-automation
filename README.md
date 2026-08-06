# YouTube Automation

This project is a Streamlit-based dashboard for repurposing long-form YouTube videos into short-form clips.

## What it does

The app can:
- download a YouTube video and its audio
- transcribe the audio with Whisper
- analyze the transcript with Gemini AI to find the most engaging highlights
- generate AI voiceovers with Edge TTS
- render vertical short clips with subtitles and combine them into a final video

## Features

- Simple web UI for entering a YouTube URL and choosing a voice
- Automatic clip generation for viral-style Shorts/Reels
- Subtitle overlays on rendered clips
- Final compilation export as an MP4 file

## Requirements

Install the required Python packages:

```powershell
C:/Python313/python.exe -m pip install streamlit yt-dlp edge-tts faster-whisper google-genai python-dotenv
```

You also need:
- Python 3.10+
- ffmpeg installed and available in your PATH
- a Gemini API key set as `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## Run the app

From the project folder, run:

```powershell
C:/Python313/python.exe -m streamlit run dashboard.py --server.headless true
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Project structure

- `dashboard.py` - main Streamlit application
- `output_clips/` - generated video and clip output files

## Notes

- The app expects a local `.env` file for secrets if you do not want to set environment variables directly.
- Generated media files are ignored by git by default.

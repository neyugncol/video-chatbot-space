---
license: mit
title: Chatbot for Video Question Answering
sdk: gradio
emoji: 📚
pinned: false
short_description: A chatbot that can answer questions about a video.
python_version: 3.12.7
sdk_version: 5.35.0
---

# Chatbot for Video Question Answering Demo

AI chatbot that can answer questions about video content. This project leverages multi-modal LLM, multi-modal RAG pipeline to process video frames, transcribe audio, and retrieval information to provide accurate answers to questions about video content.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for package and project manager
- [FFmpeg](https://ffmpeg.org/) installed and available in PATH
- [Google Gemini API key](https://aistudio.google.com/apikey) for the LLM functionality

## Installation

1. Clone this repository
   ```bash
   git clone [repository-url]
   cd VideoChatbot
   ```

2. Install dependencies using uv
   ```bash
   uv sync
   ```

3. Create a `.env` file in the project root with your API key
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the application
   ```bash
   python -m app.main
   ```

2. Access the UI through your browser (typically at http://127.0.0.1:7860)

3. Upload a video file or provide a YouTube URL and ask questions about it

4. The system will process the video (extract frames, transcribe audio), index the content, and then answer your questions

## Notes

This project is designed to be a demo and may require additional configuration for production use. The video processing and indexing can take time depending on the video length and complexity. Use a larger LLMs, embeddings, transcription models, and vector databases for better performance and accuracy.
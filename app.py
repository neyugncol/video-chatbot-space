import os
import shutil

import gradio as gr
from smolagents import ChatMessageToolCall, ActionStep, FinalAnswerStep

import utils
from agent import VideoChatbot
from configs import settings


bot = VideoChatbot(
    model=settings.CHATBOT_MODEL,
    api_base=settings.MODEL_BASE_API,
    api_key=os.environ['GEMINI_API_KEY']
)


def chat(message: dict, history: list[dict]):

    # move the file to the data directory
    message['files'] = [shutil.copy(file, settings.DATA_DIR) for file in message['files']]

    # add the input message to the history
    history.extend([{'role': 'user', 'content': {'path': file}} for file in message['files']])
    history.append({'role': 'user', 'content': message['text']})
    yield history, ''

    for step in bot.chat(message['text'], message['files']):
        match step:
            case ChatMessageToolCall():
                if step.function.name == 'download_video':
                    history.append({
                        'role': 'assistant',
                        'content': f'📥 Downloading video from {step.function.arguments["url"]}'
                    })
                elif step.function.name == 'index_video':
                    video_path = os.path.join(settings.DATA_DIR, step.function.arguments['filename'])
                    video_duration = utils.seconds_to_hms(int(utils.get_media_duration(video_path)))
                    history.append({
                        'role': 'assistant',
                        'content': f'🎥 Indexing video `{step.function.arguments["filename"]}` with length *{video_duration}* '
                                   f'to the knowledge base. This may take a while...'
                    })
                elif step.function.name == 'search_video_segments':
                    filename = os.path.basename(bot.video_rag.videos[step.function.arguments["video_id"]]['video_path'])
                    history.append({
                        'role': 'assistant',
                        'content': f'🔍 Searching video segments in `{filename}` '
                                   f'for query: *{step.function.arguments.get("text_query", step.function.arguments.get("image_query", ""))}*'
                    })
                elif step.function.name == 'read_video_segment':
                    filename = os.path.basename(bot.video_rag.videos[step.function.arguments["video_id"]]['video_path'])
                    history.append({
                        'role': 'assistant',
                        'content': f'📖 Reading video segment `{filename}` '
                                   f'from *{step.function.arguments["start_time"]}* to *{step.function.arguments["end_time"]}*'
                    })
                elif step.function.name == 'final_answer':
                    continue
                yield history, ''
            case ActionStep():
                yield history, ''
            case FinalAnswerStep():
                history.append({'role': 'assistant', 'content': step.output})
                yield history, ''


def clear_chat(chatbot):
    chatbot.clear()
    return chatbot, gr.update(value='')


def main():
    with gr.Blocks() as demo:
        gr.Markdown('# Video Chatbot Demo')
        gr.Markdown('This demo showcases a video chatbot that can process and search videos using '
                    'RAG (Retrieval-Augmented Generation). You can upload videos/images or link to YouTube videos, '
                    'ask questions, and get answers based on the video content.')
        chatbot = gr.Chatbot(type='messages', label='Video Chatbot', height=800, resizable=True)
        textbox = gr.MultimodalTextbox(
            sources=['upload'],
            file_types=['image', '.mp4'],
            show_label=False,
            placeholder='Type a message or upload an image/video...',

        )
        textbox.submit(chat, [textbox, chatbot], [chatbot, textbox])
        clear = gr.Button('Clear Chat')
        clear.click(clear_chat, [chatbot], [chatbot, textbox])

    demo.launch(debug=True)

if __name__ == '__main__':
    main()
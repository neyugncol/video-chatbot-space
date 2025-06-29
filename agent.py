import re
from typing import Generator

from smolagents import ToolCallingAgent, OpenAIServerModel, ActionStep
from PIL import Image

import tools
from configs import settings
from prompt import video_to_text_prompt
from rag import VideoRAG


class VideoChatbot:
    def __init__(
            self,
            model: str = 'gemini-2.0-flash',
            api_base: str = None,
            api_key: str = None
    ):
        self.video_rag = VideoRAG(
            video_frame_rate=settings.VIDEO_EXTRACTION_FRAME_RATE,
            audio_segment_length=settings.AUDIO_SEGMENT_LENGTH,
        )
        self.agent = ToolCallingAgent(
            tools=[
                tools.download_video,
                *tools.create_video_rag_tools(self.video_rag)
            ],
            model=OpenAIServerModel(
                model_id=model,
                api_base=api_base,
                api_key=api_key
            ),
            step_callbacks=[self._step_callback],
        )

    def chat(self, message: str, attachments: list[str] = None) -> Generator:
        """Chats with the bot, including handling attachments (images and videos).

        Args:
            message: The text message to send to the bot.
            attachments: A list of file paths for images or videos to include in the chat.

        Returns:
            A generator yielding step objects representing the bot's responses and actions.
        """

        images = []
        for filepath in attachments or []:
            if filepath.endswith(('.jpg', '.jpeg', '.png')):
                images.append(Image.open(filepath))
            if filepath.endswith('.mp4'):
                message = video_to_text_prompt(filepath) + message

        for step in self.agent.run(
            message,
            stream=True,
            reset=False,
            images=images,
        ):
            yield step

    def clear(self):
        """Clears the chatbot message history and context."""
        self.agent.state.clear()
        self.agent.memory.reset()
        self.agent.monitor.reset()
        self.video_rag.clear()

    def _step_callback(self, step: ActionStep, agent: ToolCallingAgent):
        if step.observations:
            image_index = 0
            for image_path in re.findall(r'<observation_image>(.*?)</observation_image>', step.observations):
                try:
                    image = Image.open(image_path)
                    step.observations_images.append(image)
                    step.observations = step.observations.replace(image_path, str(image_index))
                    image_index += 1
                except Exception as e:
                    print(f'Error loading image {image_path}: {e}')



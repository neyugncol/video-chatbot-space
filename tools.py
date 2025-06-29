import os

from smolagents import tool, Tool

import utils
from configs import settings
from prompt import video_to_text_prompt
from rag import VideoRAG


@tool
def download_video(url: str) -> str:
    """
    Download a video from YouTube or other supported platforms.

    Args:
        url (str): The URL of the video.

    Returns:
        str: The video information, including the filename.
    """

    try:
        filepath, info = utils.download_video(
            url,
            output_dir=settings.DATA_DIR,
            max_resolution=settings.MAX_VIDEO_RESOLUTION,
            max_fps=settings.MAX_VIDEO_FPS,
            extension=settings.VIDEO_EXTENSION
        )
    except Exception as e:
        return f'Error downloading video: {e.__class__.__name__}: {e}'

    return video_to_text_prompt(
        filepath,
        metadata={
            'URL': url,
            'Title': info.get('title', 'N/A'),
            'Channel': info.get('channel', 'N/A'),
            'Duration': info.get('duration', 'N/A'),
        }
    )


def create_video_rag_tools(video_rag: VideoRAG) -> list[Tool]:

    @tool
    def add_video(filename: str) -> str:
        """
        Add a video file to the RAG knowledge-base for further search and analysis.

        Args:
            filename (str): The video filename to add.

        Returns:
            str: The video ID if added successfully, or an error message.
        """
        try:
            video_id = video_rag.add_video(os.path.join(settings.DATA_DIR, filename))
            return f'Video added with ID: {video_id}'
        except Exception as e:
            return f'Error adding video: {e.__class__.__name__}: {e}'


    @tool
    def search_in_video(video_id: str, text_query: str = None, image_query: str = None) -> str:
        """
        Search for relevant video frames and transcripts based on text or image query. Allows searching within a specific video added to the RAG knowledge-base.
        At least one of `text_query` or `image_query` must be provided.

        Args:
            video_id (str): The ID of the video to search in. This should be the ID returned by `add_video`.
            text_query (str, optional): The text query to search for in the video transcripts.
            image_query (str, optional): The image query to search for in the video frames. This is the filename of the image.

        Returns:
            str: A message indicating the search results or an error message if the video is not found.
        """

        if not video_rag.is_video_exists(video_id):
            return f'Video with ID "{video_id}" not found in the knowledge-base. Please add the video first using `add_video` tool.'
        if not text_query and not image_query:
            return 'Please provide at least one of `text_query` or `image_query` to search in the video.'

        try:
            results = video_rag.search(
                video_id=video_id,
                text=text_query,
                image=image_query,
                limit=5
            )
        except Exception as e:
            return f'Error searching in video: {e.__class__.__name__}: {e}'

        if not results:
            return f'No results found for the given query in video ID {video_id}.'

        # build the output message
        output = f'Search results for video ID {video_id}:\n'
        for result in results:
            # include timespans, transcript segments, and frame paths in the output
            timespan_text = f'{utils.seconds_to_hms(int(result['start']))} - {utils.seconds_to_hms(int(result['end']))}'
            transcript_texts = []
            for segment in result['transcript_segments']:
                transcript_texts.append(
                    f'- {utils.seconds_to_hms(int(segment['start']), drop_hours=True)}'
                    f'-{utils.seconds_to_hms(int(segment['end']), drop_hours=True)}: {segment['text']}')
            observation_image_texts = []
            for frame_path in result['frame_paths'][::5]:  # take every 5th frame for brevity
                observation_image_texts.append(f'<observation_image>{frame_path}</observation_image>')

            output += f'''<video_segment>
Timespan: {timespan_text}
Transcript:
{'\n'.join(transcript_texts)}
Frame images: {' '.join(observation_image_texts)}
</video_segment>\n'''

        return output

    return [add_video, search_in_video]


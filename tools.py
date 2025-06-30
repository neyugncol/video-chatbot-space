import os

from smolagents import tool, Tool

import utils
from configs import settings
from prompt import video_to_text_prompt, video_segment_to_text_prompt
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
    def index_video(filename: str) -> str:
        """
        Index a video file to the RAG knowledge-base for further search and analysis.

        Args:
            filename (str): The video filename to index.

        Returns:
            str: The video ID if indexed successfully, or an error message.
        """
        try:
            video_id = video_rag.index(os.path.join(settings.DATA_DIR, filename))
            return f'Video indexed with ID: {video_id}'
        except Exception as e:
            return f'Error indexing video: {e.__class__.__name__}: {e}'


    @tool
    def search_video_segments(video_id: str, text_query: str = None, image_query: str = None) -> str:
        """
        Search for relevant video frames and transcripts based on text or image query. Allows searching within a specific video indexed to the RAG knowledge-base.
        At least one of `text_query` or `image_query` must be provided.
        The image frames of the retrieved video segments will be output at a frame rate of 1 frame per second. The order of the frames is according to the returned video segments.

        Args:
            video_id (str): The ID of the video to search in. This should be the ID returned by `index_video`.
            text_query (str, optional): The text query to search for in the video transcripts.
            image_query (str, optional): The image query to search for in the video frames. This is the filename of the image.

        Returns:
            str: A message indicating the search results or an error message if the video is not found. The output will include the timespan of the video segments, transcript segments, and images of the frames.
        """

        if not video_rag.is_video_exists(video_id):
            return f'Video with ID "{video_id}" not found in the knowledge-base. Please index the video first using `index_video` tool.'
        if not text_query and not image_query:
            return 'Please provide at least one of `text_query` or `image_query` to search in the video.'
        if image_query:
            image_query = os.path.join(settings.DATA_DIR, image_query)

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
            output += video_segment_to_text_prompt(
                start=result['start'],
                end=result['end'],
                transcript_segments=result['transcript_segments'],
                frame_paths=result['frame_paths']
            )

        return output
    
    def read_video_segment(video_id: str, start: str, end: str) -> str:
        """
        Read a specific segment of a video by its ID and time range. Use this tool when you want to read a specific segment of a video for further analysis. Don't use this tool to search for video segments, use `search_video_segments` instead. Don't read too long segments.

        Args:
            video_id (str): The ID of the video to read.
            start (str): The start time in HH:MM:SS or MM:SS format. (e.g., "00:01:30" or "01:30" for 1 minute 30 seconds)
            end (str): The end time in HH:MM:SS or MM:SS format. (e.g., "00:02:00" or "02:00" for 2 minutes)

        Returns:
            str: A message indicating the segment has been read or an error message if the video is not found. The output will include the video segment's timespan and the path to the video segment file.
        """
        if not video_rag.is_video_exists(video_id):
            return f'Video with ID "{video_id}" not found in the knowledge-base. Please index the video first using `index_video` tool.'

        # convert start and end to seconds
        start_seconds = utils.hms_to_seconds(start)
        end_seconds = utils.hms_to_seconds(end)

        try:
            result = video_rag.read(video_id, start_seconds, end_seconds)
        except Exception as e:
            return f'Error reading video segment: {e.__class__.__name__}: {e}'

        return f'''Read video segment of video ID {video_id}:
{video_segment_to_text_prompt(
    start=start_seconds,
    end=end_seconds,
    transcript_segments=result['transcript_segments'],
    frame_paths=result['frame_paths']
)}'''

    return [index_video, search_video_segments, read_video_segment]

import os

import utils


def image_to_text_prompt(image_path: str, metadata: dict = None) -> str:
    """Generate a text prompt to represent a image file with its metadata."""
    metadata = metadata or {}
    metadata_lines = '\n'.join(f'- {key}: {value}' for key, value in metadata.items())
    if metadata_lines:
        metadata_lines = '\n' + metadata_lines
    return f'''<image>
Filename: {os.path.basename(image_path)}
Metadata: {metadata_lines}
</image>
'''


def video_to_text_prompt(video_path: str, metadata: dict = None) -> str:
    """Generate a text prompt to represent a video file with its metadata."""
    metadata = metadata or {}
    metadata_lines = '\n'.join(f'- {key}: {value}' for key, value in metadata.items())
    if metadata_lines:
        metadata_lines = '\n' + metadata_lines
    return f'''<video>
Filename: {os.path.basename(video_path)}
Metadata: {metadata_lines}
</video>
'''


def video_segment_to_text_prompt(
        start: float,
        end: float,
        transcript_segments: list[dict],
        frame_paths: list[str]
) -> str:
    """Generate a text prompt to represent a video segment with its timespan, transcript segments, and frame images."""

    # include timespans
    timespan_text = f'{utils.seconds_to_hms(int(start))} - {utils.seconds_to_hms(int(end))}'

    # include transcript segments
    transcript_texts = []
    for segment in transcript_segments:
        transcript_texts.append(
            f'- {utils.seconds_to_hms(int(segment["start"]), drop_hours=True)}'
            f'-{utils.seconds_to_hms(int(segment["end"]), drop_hours=True)}: {segment["text"]}')
    transcript_lines = '\n'.join(transcript_texts)
    if transcript_lines:
        transcript_lines = '\n' + transcript_lines

    # include frame images
    image_tags = []
    for frame_path in frame_paths:
        image_tags.append(f'<image>{frame_path}</image>')
    frame_images_lines = '\n'.join(image_tags)

    return f'''<video_segment>
Timespan: {timespan_text}
Transcript: {transcript_lines}
{frame_images_lines}
</video_segment>
'''

import os


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

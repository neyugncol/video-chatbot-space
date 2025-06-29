import os


def video_to_text_prompt(video_path: str, metadata: dict = None) -> str:
    """Generate a text prompt to represent a video file with its metadata."""
    metadata = metadata or {}
    return f'''<video>
Filename: {os.path.basename(video_path)}
Metadata:
{'\n'.join(f'- {key}: {value}' for key, value in metadata.items())}
</video>
'''


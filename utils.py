import glob
import os.path
import subprocess

from yt_dlp import YoutubeDL
from pymediainfo import MediaInfo

from configs import settings


def download_video(
        url: str,
        output_dir: str = None,
        max_resolution: int = 1080,
        max_fps: float = 60,
        extension: str = 'mp4'
) -> tuple[str, dict]:
    """Download a video from YouTube or other supported sites. Returns the file path and video metadata.

    Args:
        url (str): The URL of the video.
        output_dir (str, optional): Directory to save the downloaded video. Defaults to current directory.
        max_resolution (int, optional): Maximum resolution of the video to download. Defaults to 1080.
        max_fps (float, optional): Maximum frames per second of the video to download. Defaults to 60.
        extension (str, optional): File extension for the downloaded video. Defaults to 'mp4'.

    Returns:
        tuple[str, dict]: A tuple containing the path to the downloaded video file and its metadata.
    """

    ydl_opts = {
        'format': f'bestvideo[height<={max_resolution}][fps<={max_fps}][ext={extension}]+'
                  f'bestaudio/best[height<={max_resolution}][fps<={max_fps}][ext={extension}]/best',
        'merge_output_format': extension,
        'outtmpl': f'{output_dir or "."}/%(title)s.%(ext)s',
        'noplaylist': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        ydl.download([url])
        if output_dir:
            output_path = os.path.join(output_dir, ydl.prepare_filename(info))
        else:
            output_path = ydl.prepare_filename(info)

    return output_path, info


def extract_video_frames(video_path: str, output_dir: str, frame_rate: float = 1, extension: str = 'jpg') -> list[str]:
    """Extract frames from a video file at a specified frame rate.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the extracted frames.
        frame_rate (float, optional): Frame rate for extraction. Defaults to 1 frame per second.
        extension (str, optional): File extension for the extracted frames. Defaults to 'jpg'.

    Returns:
        list[str]: A sorted list of paths to the extracted frame images.
    """
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run(
        [
            settings.FFMPEG_PATH,
            # '-v', 'quiet',
            '-i', video_path,
            '-vf', f'fps={frame_rate}',
            '-y',
            f'{output_dir or "."}/%d.{extension}'
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Get all extracted frames
    results = sorted(glob.glob(f'{output_dir or "."}/*.{extension}'),
                     key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    if not results:
        raise FileNotFoundError(f'No frames found in "{output_dir}" for video "{video_path}"')

    return results


def extract_audio(video_path: str, output_dir: str = None, extension: str = 'm4a') -> str:
    """Extract audio from a video file and save it as an M4A file.

    Args:
        video_path (str): Path to the video file.
        output_dir (str, optional): Directory to save the extracted audio. Defaults to the same directory as the video.
        extension (str, optional): File extension for the extracted audio. Defaults to 'm4a'.
    Returns:
        str: Path to the extracted audio file.
    """
    if output_dir is None:
        output_dir = os.path.dirname(video_path)

    audio_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(video_path))[0]}.{extension}')

    subprocess.run(
        [
            settings.FFMPEG_PATH,
            '-i', video_path,
            '-q:a', '0',
            '-map', 'a',
            '-y',
            audio_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Audio extraction failed: "{audio_path}" does not exist.')

    return audio_path


def split_media_file(file_path: str, output_dir: str, segment_length: int = 60) -> list[str]:
    """Split a media file into segments of specified length in seconds.

    Args:
        file_path (str): Path to the media file to be split.
        output_dir (str): Directory to save the split segments.
        segment_length (int, optional): Length of each segment in seconds. Defaults to 60 seconds.

    Returns:
        list[str]: A sorted list of paths to the split media segments.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    extension = os.path.splitext(file_path)[1]
    segment_pattern = os.path.join(output_dir, f'{base_name}_%03d.{extension}')

    subprocess.run(
        [
            settings.FFMPEG_PATH,
            '-i', file_path,
            '-c', 'copy',
            '-map', '0',
            '-segment_time', str(segment_length),
            '-f', 'segment',
            '-y',
            segment_pattern
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return sorted(glob.glob(f'{output_dir}/*{base_name}_*.{extension}'))


def get_media_duration(file_path: str) -> float:
    """Get the duration of a media file in seconds."""
    # use pymediainfo to get the duration
    media_info = MediaInfo.parse(file_path)
    for track in media_info.tracks:
        if track.track_type == 'General':
            return track.duration / 1000.0
    raise ValueError(f'Could not determine duration for file: {file_path}')


def span_iou(span1: tuple[float, float], span2: tuple[float, float]) -> float:
    """Calculate the Intersection over Union (IoU) of two spans."""
    start1, end1 = span1
    start2, end2 = span2

    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    if intersection_start >= intersection_end:
        return 0.0  # No overlap

    intersection_length = intersection_end - intersection_start
    union_length = (end1 - start1) + (end2 - start2) - intersection_length

    return intersection_length / union_length if union_length > 0 else 0.0


def seconds_to_hms(total_seconds: int, drop_hours: bool = False) -> str:
    """Convert a number of seconds to a string formatted as HH:MM:SS."""
    # Ensure we’re working with non-negative integers
    if total_seconds < 0:
        raise ValueError('total_seconds must be non-negative')

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if drop_hours and hours == 0:
        return f'{minutes:02d}:{seconds:02d}'

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def hms_to_seconds(hms: str) -> int:
    """Convert a string formatted as HH:MM:SS to total seconds."""
    parts = hms.split(':')
    if len(parts) == 2:  # MM:SS format
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS format
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError('Invalid time format. Use HH:MM:SS or MM:SS.')
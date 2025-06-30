import os.path
import uuid

import lancedb
import pyarrow as pa
from PIL import Image
from scipy.spatial import distance
from tqdm import tqdm
import spaces

import utils
from configs import settings
from embeder import MultimodalEmbedder
from transcriber import AudioTranscriber


class VideoRAG:
    """Video RAG (Retrieval-Augmented Generation) system for processing and searching video content."""

    def __init__(self, video_frame_rate: float = 1, audio_segment_length: int = 300):
        self.video_frame_rate = video_frame_rate
        self.audio_segment_length = audio_segment_length

        print('Loading embedding and audio transcription models...')
        self.embedder = MultimodalEmbedder(
            text_model=settings.TEXT_EMBEDDING_MODEL,
            image_model=settings.IMAGE_EMBEDDING_MODEL,
        )
        self.transcriber = AudioTranscriber()

        # init DB and tables
        self._init_db()

    def _init_db(self):
        print('Initializing LanceDB...')
        self.db = lancedb.connect(f'{settings.DATA_DIR}/vectordb')
        self.frames_table = self.db.create_table('frames', mode='overwrite', schema=pa.schema([
            pa.field('vector', pa.list_(pa.float32(), self.embedder.image_embedding_size)),
            pa.field('video_id', pa.string()),
            pa.field('frame_index', pa.int32()),
            pa.field('frame_path', pa.string()),
        ]))
        self.transcripts_table = self.db.create_table('transcripts', mode='overwrite', schema=pa.schema([
            pa.field('vector', pa.list_(pa.float32(), self.embedder.text_embedding_size)),
            pa.field('video_id', pa.string()),
            pa.field('segment_index', pa.int32()),
            pa.field('start', pa.float64()),
            pa.field('end', pa.float64()),
            pa.field('text', pa.string()),
        ]))

        # save video metadata
        self.videos = {}

    def is_video_exists(self, video_id: str) -> bool:
        """Check if a video exists in the RAG system by video ID.

        Args:
            video_id (str): The ID of the video to check.

        Returns:
            bool: True if the video exists, False otherwise.
        """
        return video_id in self.videos

    def get_video(self, video_id: str) -> dict:
        """Retrieve video metadata by video ID.

        Args:
            video_id (str): The ID of the video to retrieve.

        Returns:
            dict: A dictionary containing video metadata, including video path, frame directory, frame rate, and transcript segments.
        """
        if video_id not in self.videos:
            raise ValueError(f'Video with ID {video_id} not found.')
        return self.videos[video_id]

    def index(self, video_path: str) -> str:
        """Index a video file into the RAG system by extracting frames, transcribing audio, and computing embeddings.

        Args:
            video_path (str): The path to the video file to be indexed.

        Returns:
            str: A unique video ID generated for the indexed video.
        """
        # create a unique video ID
        video_id = uuid.uuid4().hex[:8]

        print(f'Indexing video "{video_path}" with ID {video_id} to the RAG system...')

        print('Extracting video frames')
        # process video frames
        frame_paths = utils.extract_video_frames(
            video_path,
            output_dir=f'{video_path}_frames',
            frame_rate=self.video_frame_rate
        )
        print('Extracting audio from video')
        # transcribe video to text
        audio_path = utils.extract_audio(video_path)
        print(f'Splitting and transcribing audio...')
        segments = []
        for i, segment_path in tqdm(enumerate(utils.split_media_file(
                audio_path,
                output_dir=f'{video_path}_audio_segments',
                segment_length=self.audio_segment_length
        )), desc='Transcribing audio'):
            for segment in self.transcriber.transcribe(segment_path)['segments']:
                segment['start'] += i * self.audio_segment_length
                segment['end'] += i * self.audio_segment_length
                segments.append(segment)
        segments = sorted(segments, key=lambda s: s['start'])

        print(f'Computing embeddings for audio transcripts and video frames...')
        transcript_embeddings, frame_embeddings = compute_embeddings(
            self.embedder,
            texts=[s['text'] for s in segments],
            images=frame_paths
        )
        # add transcripts to the database
        self.transcripts_table.add(
            [{
                'vector': transcript_embeddings[i],
                'video_id': video_id,
                'segment_index': i,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
            } for i, segment in enumerate(segments)],
        )
        print(f'Added {len(segments)} transcript segments to the database.')

        # get significant frames to reduce the number of frames
        frame_indexes = get_significant_frames(frame_embeddings, threshold=0.95)
        print(f'Found {len(frame_indexes)} significant frames out of {len(frame_embeddings)} total frames.')
        # add frames to the database
        self.frames_table.add(
            [{
                'vector': frame_embeddings[i],
                'video_id': video_id,
                'frame_index': i,
                'frame_path': frame_paths[i],
            } for i in frame_indexes]
        )
        print(f'Added {len(frame_indexes)} significant frames to the database.')

        # add video metadata to the database
        self.videos[video_id] = {
            'video_path': video_path,
            'video_duration': utils.get_media_duration(video_path),
            'frame_dir': f'{video_path}_frames',
            'video_frame_rate': self.video_frame_rate,
            'transcript_segments': segments,
        }

        print(f'Video "{video_path}" indexed with ID {video_id}.')
        return video_id

    def search(self, video_id: str, text: str = None, image: str | Image.Image = None, limit: int = 10) -> list[dict]:
        """Search for relevant video frames or transcripts based on text or image input.

        Args:
            video_id (str): The ID of the video to search in.
            text (str, optional): The text query to search for in the video transcripts.
            image (str | Image.Image, optional): The image query to search for in the video frames. If a string is provided, it should be the path to the image file.
            limit (int, optional): The maximum number of results to return. Defaults to 10.

        Returns:
            list[dict]: A list of dictionaries containing the search results, each with start and end times, distance, frame paths, and transcript segments.
        """
        video_metadata = self.get_video(video_id)
        timespans = []

        if text is not None:
            text_embedding = self.embedder.embed_texts([text], kind='query')[0]
            # search for transcripts based on text
            query = (self.transcripts_table
                    .search(text_embedding)
                    .where(f'video_id = \'{video_id}\'')
                    .limit(limit))
            for result in query.to_list():
                similarity = self.embedder.similarity(
                    [text_embedding],
                    [result['vector']],
                    pair_type='text-text'
                )[0][0]
                timespans.append({
                    'start': result['start'],
                    'end': result['end'],
                    'similarity': similarity,
                })

            # search for frames based on text
            query = (self.frames_table
                    .search(text_embedding)
                    .where(f'video_id = \'{video_id}\'')
                    .limit(limit))
            for result in query.to_list():
                similarity = self.embedder.similarity(
                    [text_embedding],
                    [result['vector']],
                    pair_type='text-image'
                )[0][0]
                start = result['frame_index'] / self.video_frame_rate
                timespans.append({
                    'start': start,
                    'end': start + 1,
                    'similarity': similarity,
                })

        if image is not None:
            image_embedding = self.embedder.embed_images([image])[0]
            # search for frames based on image
            query = (self.frames_table
                    .search(image_embedding)
                    .where(f'video_id = \'{video_id}\'')
                    .limit(limit))
            for result in query.to_list():
                similarity = self.embedder.similarity(
                    [image_embedding],
                    [result['vector']],
                    pair_type='image-image'
                )[0][0]
                start = result['frame_index'] / self.video_frame_rate
                timespans.append({
                    'start': start,
                    'end': start + 1,
                    'similarity': similarity,
                })
            # search for transcripts based on image
            query = (self.transcripts_table
                    .search(image_embedding)
                    .where(f'video_id = \'{video_id}\'')
                    .limit(limit))
            for result in query.to_list():
                similarity = self.embedder.similarity(
                    [image_embedding],
                    [result['vector']],
                    pair_type='text-image'
                )[0][0]
                timespans.append({
                    'start': result['start'],
                    'end': result['end'],
                    'similarity': similarity,
                })

        # merge nearby timespans
        timespans = merge_searched_timespans(timespans, threshold=5)
        # sort timespans by distance
        timespans = sorted(timespans, key=lambda x: x['similarity'], reverse=True)
        # limit to k results
        timespans = timespans[:limit]

        for timespan in timespans:
            # extend timespans to at least 5 seconds
            duration = timespan['end'] - timespan['start']
            if duration < 5:
                timespan['start'] = max(0, timespan['start'] - (5 - duration) / 2)
                timespan['end'] = timespan['start'] + 5
            # add frame paths
            timespan['frame_paths'] = []
            for frame_index in range(
                    int(timespan['start'] * self.video_frame_rate),
                    int(timespan['end'] * self.video_frame_rate)
            ):
                timespan['frame_paths'].append(os.path.join(video_metadata['frame_dir'], f'{frame_index + 1}.jpg'))
            # add transcript segments
            timespan['transcript_segments'] = []
            for segment in video_metadata['transcript_segments']:
                if utils.span_iou((segment['start'], segment['end']),
                                  (timespan['start'], timespan['end'])) > 0:
                    timespan['transcript_segments'].append(segment)

        return timespans

    def read(self, video_id: str, start: float, end: float) -> dict:
        """Read a segment of the video by its ID and time range.

        Args:
            video_id (str): The ID of the video to read.
            start (float): The start time of the segment in seconds.
            end (float): The end time of the segment in seconds.

        Returns:
            dict: A dictionary containing the video segment metadata, including start and end times, frame paths, and transcript segments.
        """
        video_metadata = self.get_video(video_id)

        if start > video_metadata['video_duration'] or end > video_metadata['video_duration']:
            raise ValueError(f'Start ({start}) or end ({end}) time exceeds video duration ({video_metadata["video_duration"]}).')

        timespan = {
            'start': start,
            'end': end,
            'frame_paths': [],
            'transcript_segments': []
        }

        # add frame paths
        for frame_index in range(
                int(start * self.video_frame_rate),
                int(end * self.video_frame_rate)
        ):
            timespan['frame_paths'].append(os.path.join(video_metadata['frame_dir'], f'{frame_index + 1}.jpg'))

        # add transcript segments
        for segment in video_metadata['transcript_segments']:
            if utils.span_iou((segment['start'], segment['end']),
                              (start, end)) > 0:
                timespan['transcript_segments'].append(segment)

        return timespan


    def clear(self):
        """Clear the RAG system by dropping all tables and resetting video metadata."""
        self._init_db()


@spaces.GPU
def compute_embeddings(
        embedder: MultimodalEmbedder,
        texts: list[str],
        images: list[str | Image.Image]
) -> tuple[list[list[float]], list[list[float]]]:
    print(f'Computing embeddings for {len(texts)} texts...')
    text_embeddings = embedder.embed_texts(texts, kind='document', device='cuda')
    print(f'Computing embeddings for {len(images)} images...')
    image_embeddings = embedder.embed_images(images, device='cuda')

    return text_embeddings, image_embeddings


def get_significant_frames(frame_embeddings: list[list[float]], threshold: float = 0.8) -> list[int]:
    """Select significant frames by comparing embeddings."""
    selected_frames = []
    current_frame = 0
    for i, embedding in enumerate(frame_embeddings):
        similarity = 1 - distance.cosine(frame_embeddings[current_frame], embedding)
        if similarity < threshold:
            selected_frames.append(current_frame)
            current_frame = i

    selected_frames.append(current_frame)

    return selected_frames


def merge_searched_timespans(timespans: list[dict], threshold: float) -> list[dict]:
    """Merge timespans if the gap between them is less than or equal to threshold."""
    if not timespans:
        return []

    # Sort spans by start time
    sorted_spans = sorted(timespans, key=lambda s: s['start'])

    merged_spans = []
    current_span = sorted_spans[0].copy()

    for next_span in sorted_spans[1:]:
        gap = next_span['start'] - current_span['end']
        if gap <= threshold:
            # Extend the current span’s end if needed
            current_span['end'] = max(current_span['end'], next_span['end'])
            current_span['similarity'] = max(current_span['similarity'], next_span['similarity'])
        else:
            # No merge push current and start a new one
            merged_spans.append(current_span)
            current_span = next_span.copy()

    # Add the last span
    merged_spans.append(current_span)
    return merged_spans
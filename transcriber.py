import time
from typing import Any

from google import genai
from google.genai import types

import utils


class AudioTranscriber:
    """A class to transcribe audio files"""

    SYSTEM_INSTRUCTION = '''You are an advanced audio transcription model. Your task is to accurately transcribe provided audio input into a structured JSON format.

**Output Format Specification:**

Your response MUST be a valid JSON object with the following structure:

```json
{
  "segments": [
    {
      "text": "The transcribed text for the segment.",
      "start": "The start time of the segment in seconds.",
      "end": "The end time of the segment in seconds.",
      "speaker": "The speaker ID for the segment."
    }
  ],
  "language": "The language of the transcribed text in ISO 639-1 format."
}
```

**Detailed Instructions and Rules:**

1. Segments:
- A "segment" is defined as a continuous section of speech from a single speaker include multiple sentences or phrases.
- Each segment object MUST contain `text`, `start`, `end`, and `speaker` fields.
- `text`: The verbatim transcription of the speech within that segment.
- `start`: The precise start time of the segment in seconds, represented as a integer number (e.g., 1, 5)
- `end`: The precise end time of the segment in seconds, represented as a integer number (e.g., 2, 6)
- `speaker`: An integer representing the speaker ID.
  + Speaker IDs start at `0` for the first detected speaker.
  + The speaker ID MUST increment by 1 each time a new, distinct speaker is identified in the audio. Do not reuse speaker IDs within the same transcription.
  + If the same speaker talks again after another speaker, they retain their original speaker ID.
  + **Segment Splitting Rule**: A segment for the same speaker should only be split if there is a period of silence lasting more than 5 seconds or have length longer than 30 seconds. Otherwise, continuous speech from the same speaker, even with short pauses, should remain within a single segment.

2. Language:
- `language`: A two-letter ISO 639-1 code representing the primary language of the transcribed text (e.g., "en" for English, "es" for Spanish, "fr" for French).
-  If multiple languages are detected in the audio, you MUST select and output only the ISO 639-1 code for the primary language used throughout the audio.
'''

    RESPONSE_SCHEMA = {
        'type': 'object',
        'properties': {
            'segments': {
                'type': 'array',
                "description": 'A list of transcribed segments from the audio file.',
                'items': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'The transcribed text for the segment.'
                        },
                        'start': {
                            'type': 'integer',
                            'description': 'The start time of the segment in seconds.'
                        },
                        'end': {
                            'type': 'integer',
                            'description': 'The end time of the segment in seconds.'
                        },
                        'speaker': {
                            'type': 'integer',
                            'description': 'The speaker ID for the segment.'
                        }
                    },
                    'required': ['text', 'start', 'end', 'speaker'],
                    'propertyOrdering': ['text', 'start', 'end', 'speaker']
                },
            },
            'language': {
                'type': 'string',
                'description': 'The language of the transcribed text in ISO 639-1 format.',
            }
        },
        'required': ['segments', 'language'],
        'propertyOrdering': ['segments', 'language']
    }

    def __init__(self, model: str = 'gemini-2.0-flash', api_key: str = None):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """Transcribe an audio file from the given path.

        Args:
            audio_path (str): The path to the audio file to be transcribed.

        Returns:
            dict[str, Any]: The transcription result.
            ```{
                "segments": [
                    {
                        "text": "Transcribed text",
                        "start": 0.0,
                        "end": 5.0,
                        "speaker": 0
                    }
                ],
                "language": "en"
            }```
        """

        uploaded_file = self.client.files.upload(file=audio_path)
        while uploaded_file.state != 'ACTIVE':
            time.sleep(1)
            uploaded_file = self.client.files.get(name=uploaded_file.name)
            if uploaded_file.state == 'FAILED':
                raise ValueError('Failed to upload the audio file')

        audio_duration = utils.get_media_duration(audio_path)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[uploaded_file, f'Audio duration: {int(audio_duration)} seconds'],
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_INSTRUCTION,
                temperature=0.2,
                response_mime_type='application/json',
                response_schema=self.RESPONSE_SCHEMA,
            )
        )

        if response.parsed is None:
            raise ValueError('Failed to transcribe the audio file')

        return response.parsed  # type: ignore
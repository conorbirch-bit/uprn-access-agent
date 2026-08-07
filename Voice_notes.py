from openai import OpenAI


def transcribe_audio(audio_file, api_key):
    client = OpenAI(api_key=api_key)

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=(
            "site_note.wav",
            audio_file.getvalue(),
            "audio/wav",
        ),
        language="en",
    )

    return transcript.text.strip()

"""Guards for video re-encoding that preserves an older audio/text payload."""

def validate_preserved_audio_range(row, *, fps=24):
    video_range=row.get('source_time_range')
    audio_range=row.get('audio_source_time_range')
    if row.get('source_metadata_origin')=='native_provenance' and audio_range is None:
        raise ValueError('Native source requires verified audio_source_time_range before reusing audio: '
                         'the original preprocessing center-crops; zero video start is unsafe')
    if audio_range is not None:
        if video_range is None or len(audio_range)!=2 or len(video_range)!=2:
            raise ValueError('Audio/video source time ranges must each contain two endpoints')
        if any(abs(float(a)-float(v))>1/fps+1e-6 for a,v in zip(audio_range,video_range)):
            raise ValueError('Preserved audio and rebuilt video refer to different source time ranges')

import pytest
from h3dmd.cache_alignment import validate_preserved_audio_range

def test_native_zero_start_cannot_silently_reuse_centered_audio():
    with pytest.raises(ValueError,match='verified audio_source_time_range'):
        validate_preserved_audio_range(dict(source_metadata_origin='native_provenance',source_time_range=[0,5.166667]))

def test_corrected_native_range_can_preserve_audio():
    validate_preserved_audio_range(dict(source_metadata_origin='native_provenance',source_time_range=[1.2975,6.464167],audio_source_time_range=[1.2975,6.4225]))

def test_known_time_mismatch_is_rejected():
    with pytest.raises(ValueError,match='different source time ranges'):
        validate_preserved_audio_range(dict(source_time_range=[0,5.166667],audio_source_time_range=[1.2975,6.4225]))

def test_validated_legacy_source_keeps_existing_range_contract():
    validate_preserved_audio_range(dict(source_metadata_origin='validated_legacy',source_time_range=[15.687,20.812]))

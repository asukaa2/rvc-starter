"""Audio splitting / restoration utilities used by the inference pipeline.

The :func:`cut` function slices a long input waveform at silent gaps so
that each chunk can be converted independently; :func:`restore` reassembles
the converted chunks back into a single waveform that has the same length
as the original input.
"""

import numpy as np
import librosa


def cut(audio, sample_rate, db_thresh=-60, min_interval=500):
    """Split ``audio`` at silent gaps.

    Parameters
    ----------
    audio : np.ndarray
        Mono 1-D audio signal.
    sample_rate : int
        Sample rate of ``audio``.
    db_thresh : float, default ``-60``
        Absolute dB threshold (relative to full-scale) below which a frame is
        treated as silent.
    min_interval : float, default ``500``
        Minimum silent gap length, in milliseconds, that may be used as a
        split point.  Shorter silent gaps are kept inside the surrounding
        chunk so we don't introduce audible re-onsets every breath.

    Returns
    -------
    list[tuple[np.ndarray, int, int]]
        One ``(waveform, start_sample, end_sample)`` tuple per chunk.  The
        chunks are non-overlapping and together cover the entire input.
    """
    if audio.ndim != 1:
        audio = librosa.to_mono(audio.T)

    total_len = audio.shape[0]
    if total_len == 0:
        return [(audio, 0, 0)]

    # librosa.effects.split() uses `top_db` as a positive number of dB below
    # the reference peak.  Our caller passes an absolute dBFS threshold so we
    # convert it accordingly.
    top_db = max(1.0, float(-db_thresh))
    min_silence_samples = int(sample_rate * float(min_interval) / 1000)

    intervals = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=2048,
        hop_length=512,
    )

    if intervals.size == 0:
        return [(audio, 0, total_len)]

    chunks = []
    # Merge adjacent non-silent intervals whose silent gap is shorter than
    # ``min_interval`` ms — this keeps natural pauses inside a chunk instead
    # of starting a new chunk mid-syllable.
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        if gap < min_silence_samples:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    for start, end in merged:
        start = max(0, int(start))
        end = min(total_len, int(end))
        if end <= start:
            continue
        chunks.append((audio[start:end].copy(), start, end))

    if not chunks:
        chunks = [(audio, 0, total_len)]

    return chunks


def restore(chunks, total_len, dtype=np.float32):
    """Re-assemble converted chunks into a single waveform.

    Parameters
    ----------
    chunks : list[tuple[int, int, np.ndarray]]
        Output of :meth:`Pipeline.pipeline` per chunk: each entry is
        ``(start_sample, end_sample, converted_audio)``.
    total_len : int
        Length of the original audio in samples.  Silent gaps between chunks
        are zero-padded so the output has this length.
    dtype : np.dtype
        Output dtype — taken from the first converted chunk in
        :meth:`VoiceConverter.convert_audio`.

    Returns
    -------
    np.ndarray
        Mono 1-D waveform of length ``total_len``.
    """
    audio_output = np.zeros(int(total_len), dtype=dtype)

    for entry in chunks:
        start, end, chunk_audio = entry
        start = max(0, int(start))
        end = min(int(total_len), int(end))
        if end <= start:
            continue

        chunk_audio = np.asarray(chunk_audio)
        take = end - start
        if chunk_audio.shape[0] >= take:
            audio_output[start:end] = chunk_audio[:take]
        else:
            # If the converted chunk is shorter than the slot, copy what we
            # have and leave the trailing silence as zeros.
            slot_end = start + chunk_audio.shape[0]
            if slot_end > end:
                slot_end = end
            audio_output[start:slot_end] = chunk_audio[: slot_end - start]

    return audio_output

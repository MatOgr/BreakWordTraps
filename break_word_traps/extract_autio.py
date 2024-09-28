from pathlib import Path
import subprocess

import imageio_ffmpeg

FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()


def extract(
    input_video: Path,
    output_audio: Path,
):
    # Get last suffix and remove the dot
    out_format = output_audio.suffixes[-1][1:]
    command = [
        FFMPEG_BINARY,
        "-i",
        str(input_video.absolute()),
        "-ss",
        "0",
        "-f",
        out_format,
        "-y",
        str(output_audio.absolute()),
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception(
            f"FFMPEG failed with error: {result.stdout.decode('utf-8')} {result.stderr.decode('utf-8')}"
        )
    return output_audio
    # extract_audio(
    #     input_path=str(input_video.absolute()),
    #     output_path=str(output_audio.absolute()),
    #     output_format=out_format,
    # )

import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile

VALID_EXTENSIONS = {".nii", ".nii.gz"}
MODALITY_KEYS = ("t1", "t1ce", "t2", "flair")


def validate_nifti_extension(filename: str) -> bool:
    name = filename.lower()
    return name.endswith(".nii.gz") or name.endswith(".nii")


async def save_upload_to_tempdir(
    files: dict[str, UploadFile],
) -> tuple[Path, dict[str, Path]]:
    """Write uploaded files to a temp directory and return (tmpdir, paths_dict)."""
    tmpdir = Path(tempfile.mkdtemp(prefix="neurograde_"))
    paths: dict[str, Path] = {}
    for modality, upload in files.items():
        dest = tmpdir / f"{modality}_{upload.filename}"
        with open(dest, "wb") as f:
            while chunk := await upload.read(1024 * 1024):
                f.write(chunk)
        paths[modality] = dest
    return tmpdir, paths


def cleanup_tempdir(tmpdir: Path) -> None:
    if tmpdir.exists():
        shutil.rmtree(tmpdir, ignore_errors=True)

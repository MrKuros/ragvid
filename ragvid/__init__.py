"""ragvid — describe the vibe, get the grade.

The public API is `Project`: one clip, its grade history, and the artifacts
derived from them. Build a UI against this, not against the modules underneath.

    from ragvid import Project

    p = Project.create("clip.mp4", root="~/grades/clip")
    p.plan_from_vibe("gloomy")
    p.refine("less blue")
    p.export("out.mp4", progress=print)

Everything below `Project` is a pure layer with no I/O policy of its own:

    spec      GradeSpec -- the 14 numbers, and the math that applies them
    probe     clip -> ClipStats (sampled frames, median statistics)
    match     (source, reference) -> GradeSpec, closed form, no model
    vibe      words -> GradeSpec, via a provider
    refine    (GradeSpec, words) -> GradeSpec, via a provider
    lut       GradeSpec -> .cube
    render    ffmpeg: contact-sheet previews and full exports
    session   persistence
    platform  the four things Linux, macOS and Windows disagree about
    errors    the typed failures a front end should branch on
"""

from .errors import (
    FFmpegError,
    FFmpegNotFound,
    InputError,
    NoGrade,
    ProviderError,
    ProviderNotConfigured,
    RagvidError,
    RateLimited,
    SessionCorrupt,
    SessionNotFound,
)
from .project import Project, available_providers
from .spec import RGB, GradeSpec

__version__ = "0.1.0"

__all__ = [
    "Project",
    "GradeSpec",
    "RGB",
    "available_providers",
    # errors, so a front end can import everything it branches on from one place
    "RagvidError",
    "InputError",
    "SessionNotFound",
    "SessionCorrupt",
    "NoGrade",
    "FFmpegError",
    "FFmpegNotFound",
    "ProviderError",
    "ProviderNotConfigured",
    "RateLimited",
    "__version__",
]

"""cric - cricket net-session video analytics.

Modules:
    video     - read frames from a clip and write an annotated clip
    pose      - detect the batsman's body skeleton (YOLOv8-pose)
    objects   - detect the ball and bat (YOLOv8 detection)
    annotate  - draw skeletons / boxes onto frames
    pipeline  - tie it all together: video in -> annotated video + data out
"""

__version__ = "0.1.0"

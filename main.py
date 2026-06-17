"""Singularity Portraits — command-line entry point.

Examples
--------
Live webcam (Phase 1, the real thing — needs a camera and face_recognition)::

    python main.py --source webcam

Camera-free demo that renders a moving constellation of synthetic identities,
and records it to a file (works anywhere, including headless servers)::

    python main.py --source synthetic --headless --record out.mp4 --max-frames 300

Run against a folder of face photos::

    python main.py --source images --images-dir ./faces
"""

from __future__ import annotations

import argparse

from singularity.app import App, AppConfig


def build_source(args):
    from singularity import sources

    if args.source == "webcam":
        return sources.WebcamSource(index=args.camera, width=1280, height=720)
    if args.source == "video":
        return sources.VideoFileSource(args.video, loop=args.loop)
    if args.source == "images":
        return sources.ImageDirSource(args.images_dir, repeat=args.repeat)
    return sources.SyntheticSource(
        width=args.width, height=args.height, num_frames=args.max_frames
    )


def build_detector(args):
    from singularity.identity import detector

    if args.source == "synthetic":
        return detector.SyntheticDetector(num_personas=args.personas)
    return detector.FaceRecognitionDetector(model=args.fr_model)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Singularity Portraits installation runner")
    p.add_argument(
        "--source",
        choices=["webcam", "video", "images", "synthetic"],
        default="synthetic",
        help="Where frames come from (default: synthetic, needs no camera).",
    )
    p.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    p.add_argument("--video", help="Path to a video file (for --source video).")
    p.add_argument("--loop", action="store_true", help="Loop the video file.")
    p.add_argument("--images-dir", help="Folder of stills (for --source images).")
    p.add_argument("--repeat", type=int, default=1, help="Times to repeat the image folder.")
    p.add_argument("--personas", type=int, default=3, help="Synthetic faces to simulate.")
    p.add_argument("--fr-model", choices=["hog", "cnn"], default="hog", help="Detector model.")

    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--threshold", type=float, default=0.6, help="Identity match distance.")
    p.add_argument("--headless", action="store_true", help="Render without a window.")
    p.add_argument("--side-by-side", action="store_true", help="Show visualizer and camera feed side by side.")
    p.add_argument("--style", choices=["classic", "vortex"], default="classic", help="Visual style.")
    p.add_argument("--max-frames", type=int, help="Stop after N frames.")
    p.add_argument("--record", help="Write output to this MP4 path.")
    p.add_argument(
        "--registry",
        help="Persist the identity registry (biometric data!) to this JSON path "
        "across runs. Omit to keep everything in memory only.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.width is None:
        args.width = 640 if args.side_by_side else 1280
    if args.height is None:
        args.height = 480 if args.side_by_side else 720
    source = build_source(args)
    detector = build_detector(args)
    config = AppConfig(
        width=args.width,
        height=args.height,
        headless=args.headless,
        fps=args.fps,
        threshold=args.threshold,
        registry_path=args.registry,
        max_frames=args.max_frames,
        record_path=args.record,
        side_by_side=args.side_by_side,
        style=args.style,
    )
    App(source, detector, config).run()


if __name__ == "__main__":
    main()

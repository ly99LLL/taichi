from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="演卦 Python demo")
    parser.add_argument("--input", choices=("mouse", "camera", "script"), default="mouse")
    parser.add_argument("--particles", type=int, default=3600)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview", action="store_true", help="Render a scripted preview grid to an image.")
    parser.add_argument("--frames", type=int, default=620, help="Frames to simulate in preview mode.")
    parser.add_argument("--output", default="preview.png", help="Preview image output path.")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-bloom", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        args.input = "script"
        args.no_audio = True

    from bagua.app import BaguaApp
    from bagua.config import Config

    config = Config(
        width=args.width,
        height=args.height,
        dark_particle_count=args.particles,
        audio_enabled=not args.no_audio,
        bloom_enabled=not args.no_bloom,
    )
    app = BaguaApp(config=config, input_mode=args.input)
    if args.preview:
        app.render_preview(args.output, args.frames)
    else:
        app.run()


if __name__ == "__main__":
    main()

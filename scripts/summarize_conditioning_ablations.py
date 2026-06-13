"""Compatibility entry point for canonical conditioning summaries."""

if __package__:
    from scripts.byte.summarize_conditioning_ablations import *  # noqa: F403
else:
    from byte.summarize_conditioning_ablations import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405

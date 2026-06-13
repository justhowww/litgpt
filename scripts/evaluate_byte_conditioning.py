"""Compatibility entry point for canonical byte conditioning evaluation."""

if __package__:
    from scripts.byte.evaluate_conditioning import *  # noqa: F403
else:
    from byte.evaluate_conditioning import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405

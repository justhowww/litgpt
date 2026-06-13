"""Compatibility entry point for the canonical byte data debugger."""

if __package__:
    from scripts.byte.debug_data import *  # noqa: F403
else:
    from byte.debug_data import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405

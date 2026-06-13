"""Compatibility entry point for canonical byte slice inspection."""

if __package__:
    from scripts.byte.inspect_slice_lengths import *  # noqa: F403
else:
    from byte.inspect_slice_lengths import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405

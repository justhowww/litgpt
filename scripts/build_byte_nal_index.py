"""Compatibility entry point for the canonical byte NAL index builder."""

if __package__:
    from scripts.byte.build_nal_index import *  # noqa: F403
else:
    from byte.build_nal_index import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405

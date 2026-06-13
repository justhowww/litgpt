"""Compatibility entry point for the canonical byte training command."""

if __package__:
    from scripts.byte.train import main
else:
    from byte.train import main


if __name__ == "__main__":
    main()

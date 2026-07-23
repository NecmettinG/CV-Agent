"""``python -m cv_agent`` -> the interactive terminal UI (see cv_agent/tui.py)."""

from cv_agent.tui import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")

"""Local script to try the chat agent directly, without the API/client."""

from dotenv import load_dotenv

load_dotenv()

from agents.chat import answer  # noqa: E402  (must load .env before this import)


def main() -> None:
    result = answer("Hello! Who are you and what can you help me with?", thread_id="local-test")
    print(result.text)


if __name__ == "__main__":
    main()

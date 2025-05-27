import argparse
import sys

from dotenv import load_dotenv

from relepaper.config.settings import load_settings
from relepaper.containers.core import configure_container

# Загружаем переменные окружения в самом начале
load_dotenv()


def main(args):
    """Запускает приложение."""
    # Загружаем настройки
    settings = load_settings()

    # Конфигурируем контейнер с настройками
    container = configure_container(settings)
    if args.gradio:
        presenter = container.resolve("gradio_presenter")
    else:
        presenter = container.resolve("console_presenter")
    presenter.run()


def cli():
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Relepaper")
    parser.add_argument("--gradio", action="store_true", default=False)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()

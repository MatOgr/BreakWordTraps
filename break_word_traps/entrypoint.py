from typer import Typer, Option, BadParameter
from typing import Optional
import uvicorn


from break_word_traps.endpoints import server_app

cli = Typer()


@cli.command()
def run_server(
    host: str = "127.0.0.1",
    port: int = 8998,
    api_key: Optional[str] = Option(None, envvar="API_KEY"),
):
    if not api_key:
        raise BadParameter("API_KEY must be defined")

    server = server_app(api_key=api_key)

    uvicorn.run(
        server.app,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    cli()

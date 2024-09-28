from typer import Typer, Option, BadParameter
from typing import Optional
import uvicorn

from break_word_traps.endpoints import server_app
from break_word_traps.utils.service_types import ServiceType

cli = Typer()


@cli.command()
def run_backend(
    asr_server_address: str,
    fer_server_address: str,
    host: str = "127.0.0.1",
    port: int = 8998,
    api_key: Optional[str] = Option(None, envvar="API_KEY"),
):
    if not api_key:
        raise BadParameter("API_KEY must be defined")

    server = server_app(service_type=ServiceType.MAIN, api_key=api_key)
    uvicorn.run(
        server.app,
        host=host,
        port=port,
    )


@cli.command()
def run_subservice(
    service_type: ServiceType,
    host: str = "127.0.0.1",
    port: int = 8998,
    api_key: Optional[str] = Option(None, envvar="API_KEY"),
    llm_server_address: Optional[str] = None,
):
    if not api_key:
        raise BadParameter("API_KEY must be defined")

    server = server_app(service_type=service_type, api_key=api_key)
    uvicorn.run(
        server.app,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    cli()

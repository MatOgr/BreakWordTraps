from typer import Typer
import uvicorn

from break_word_traps.endpoints import server_app
from break_word_traps.utils.service_types import ServiceType

cli = Typer()


@cli.command()
def run_server(
    host: str = "127.0.0.1",
    port: int = 8998,
    service_type: ServiceType = ServiceType.MAIN,
):
    server = server_app(service_type=service_type)

    uvicorn.run(
        server.app,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    cli()

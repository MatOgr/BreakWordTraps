from typer import Typer
import uvicorn


from break_word_traps.endpoints import server_app

cli = Typer()


@cli.command()
def run_server(
    host: str = "127.0.0.1",
    port: int = 8998,
):
    server = server_app()

    uvicorn.run(
        server.app,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    cli()


import argparse
from __future__ import annotations
from gtgh_team3_compliance_assistant.logger.Logger import log

def runAPI(args):
    port = args.port;
    host = args.host;
    dev_env = args.dev;
    log.info(f"Attempting to run API on host {host}, port {port}, as dev {'in development mode' if(dev_env) else 'in production mode'}")

    # All API imports
    import uvicorn
    uvicorn.run("gtgh_team3_compliance_assistant.api.run:app", host=host, port=port, reload=dev_env)

def runIngestion(args):
    from gtgh_team3_compliance_assistant.ingestion.run import run_ingestion
    run_ingestion(args)

def runStorage(args):
    from gtgh_team3_compliance_assistant.storing.run import run_storage
    run_storage(args)

def runSearch(args):
    from gtgh_team3_compliance_assistant.storing.run import run_search
    run_search(args.question)

def runEmbed(args):
    from gtgh_team3_compliance_assistant.embedding.run import run_embed
    run_embed(args.source, args.limit_chunks, args.save_embeds, args.destination)

def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to run API or Ingestion paths."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available subcommands"
    )
    # --- API Subcommand Setup ---
    parser_api = subparsers.add_parser("api", help="Run the API server")
    parser_api.add_argument(
        "-p", "--port", type=int, default=8000, help="Port to listen on"
    )
    parser_api.add_argument(
        "-H", "--host", type=str, default="localhost", help="Host ip address"
    )
    parser_api.add_argument(
        "-d", "--dev", "--development", action="store_true", help="Run API Server in development environment"
    )
    parser_api.set_defaults(func=runAPI)

    # --- Ingestion Subcommand Setup ---
    parser_ingestion = subparsers.add_parser(
        "ingest", help="Run the ingestion pipeline"
    )
    parser_ingestion.add_argument(
        "-s", "--source", required=False, help="Path to data source file"
    )
    parser_ingestion.add_argument(
        "-r", "--recreate-index", action="store_true", help="Force delete and recreate the Azure index before uploading"
    )
    parser_ingestion.set_defaults(func=runIngestion)

    # --- Store Subcommand Setup ---
    parser_ingestion = subparsers.add_parser(
        "store", help="Store JSON document data in Azure"
    )
    parser_ingestion.add_argument(
        "-s", "--source", required=True, help="Path to data source file"
    )
    parser_ingestion.add_argument(
        "-l", "--limit-chunks", action="store_true", help="Only ingest the first chunk for testing purposes"
    )
    parser_ingestion.add_argument(
        "-r", "--recreate-index", action="store_true", help="Force delete and recreate the Azure index before uploading"
    )
    parser_ingestion.set_defaults(func=runStorage)

    # --- Search Subcommand Setup ---
    parser_ingestion = subparsers.add_parser(
        "search", help="Search a query in Azure Storage"
    )
    parser_ingestion.add_argument(
        "-q", "--question", required=True, help="Question to search about"
    )
    parser_ingestion.set_defaults(func=runSearch)

    # --- Embed Subcommand Setup ---
    parser_ingestion = subparsers.add_parser(
        "embed", help="Embed a chunk saved as JSON"
    )
    parser_ingestion.add_argument(
        "-s", "--source", required=True, help="Path to data source file"
    )
    parser_ingestion.add_argument(
        "-l", "--limit-chunks", action="store_true", help="Only embed the first chunk for testing purposes"
    )
    parser_ingestion.add_argument(
        "-se", "--save-embeds", action="store_true", help="Save the generated embeds into the destination file"
    )
    parser_ingestion.add_argument(
        "-d", "--destination", required=False, help="Path to data destination file"
    )
    parser_ingestion.set_defaults(func=runEmbed)

    args = parser.parse_args()
    if args.save_embeds and not args.destination:
        parser_ingestion.error("The --destination (-d) argument is required when --save-embeds is enabled.")
    args.func(args)

if __name__ == "__main__":
    main()
import argparse

from pipelines.ingestion import ingestion
from pipelines.ranking import run_ranking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--rank", action="store_true")
    parser.add_argument("--run", action="store_true", help="Run ingest + rank")
    args = parser.parse_args()

    if args.run:
        ingestion()
        run_ranking()
        return

    if args.ingest:
        ingestion()
    if args.rank:
        run_ranking()


if __name__ == "__main__":
    main()

uv sync /////

uv run -m scripts.ingest /////

uv run uvicorn gtgh_team3_compliance_assistant.main:app --reload /////

run databricks api `uv run uvicorn gtgh_team3_compliance_assistant.main_databricks:app --reload`

## Prerequisites

Ensure your virtual environment is activated and dependencies are installed:

```bash
uv sync
```

## USAGE

The main script uses subcommands to determine execution paths. You can view the global help menu at any time:

```bash
uv run -m gtgh_team3_compliance_assistant.main --help
```

### Running API Server

Basic Usage:

```bash
uv run -m gtgh_team3_compliance_assistant.main api
```

Advanced Usage:

```bash
uv run -m gtgh_team3_compliance_assistant.main api -p 5000 -H 0.0.0.0 --dev
```

Available Flags:

|      Long Flag      | Short Flag | Type |  Default  |
| :-----------------: | :--------: | :--: | :-------: |
|       --port        |     -p     | int  |   8000    |
|       --host        |     -H     | str  | localhost |
| --dev --development |     -d     | flag |   False   |

View help menu for the api command:

```bash
uv run -m gtgh_team3_compliance_assistant.main api --help
```

### Running Data Ingestion Pipeline

Basic Usage:

```bash
uv run -m gtgh_team3_compliance_assistant.main ingest
```

Advanced Usage:

```bash
uv run -m gtgh_team3_compliance_assistant.main ingest --source ./data/custom_source.json
```

Available Flags:

|    Long Flag     | Short Flag | Type | Required |
| :--------------: | :--------: | :--: | :------: |
|     --source     |     -s     | str  |    No    |
| --recreate-index |     -r     | bool |    No    |

View help menu for the ingestion command:

```bash
uv run -m gtgh_team3_compliance_assistant.main ingest --help
```

### Running Azure Store Functionality

```bash
uv run -m gtgh_team3_compliance_assistant.main store --source data/chunks/json_name.json
```

Available Flags:

|    Long Flag     | Short Flag | Type | Required |
| :--------------: | :--------: | :--: | :------: |
|     --source     |     -s     | str  |   Yes    |
|  --limit-chunks  |     -l     | bool |    No    |
| --recreate-index |     -r     | bool |    No    |

### Running Azure Search Functionality

```bash
uv run -m gtgh_team3_compliance_assistant.main search --question "What is the document about"
```

Available Flags:

| Long Flag  | Short Flag | Type |
| :--------: | :--------: | :--: |
| --question |     -q     | str  |

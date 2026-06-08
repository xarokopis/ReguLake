# TODO: needs fixing before use
# - replace `requests` with `httpx` (not in deps)
# - update config imports (RAW_DIR, DATA_DIR_PDF → PDF_DIR)
# - consolidate async/sync approach

class Downloader:
    def __init__(self, out_dir):
        raise NotImplementedError("Downloader needs fixing — see TODOs above")

    async def download(self, url: str, filename: str):
        raise NotImplementedError

    def downloadFromEU(self, celex_id: str, language: str = "EN"):
        raise NotImplementedError

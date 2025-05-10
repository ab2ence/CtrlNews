import concurrent.futures
from typing import List, Dict

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from trafilatura import extract

class WebPageHelper:
    """Helper class to process web pages.

    Acknowledgement: Part of the code is adapted from https://github.com/stanford-oval/WikiChat project.
    """

    def __init__(self, min_char_count: int = 150, snippet_chunk_size: int = 1000, max_thread_num: int = 10):
        """
        Args:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            max_thread_num: Maximum number of threads to use for concurrent requests (e.g., downloading webpages).
        """
        self.httpx_client = httpx.Client(verify=False)
        self.min_char_count = min_char_count
        self.max_thread_num = max_thread_num
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=snippet_chunk_size,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                ".",
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                ",",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                " ",
                "\u200B",  # Zero-width space
                "",
            ],
        )

    def download_webpage(self, url: str):
        try:
            res = self.httpx_client.get(url, timeout=4)
            if res.status_code >= 400:
                res.raise_for_status()
            return res.content
        except httpx.HTTPError as exc:
            print(f"Error while requesting {exc.request.url!r} - {exc!r}")
            return None

    def urls_to_articles(self, urls: List[str]) -> Dict:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            htmls = list(executor.map(self.download_webpage, urls))

        articles = {}

        for h, u in zip(htmls, urls):
            if h is None:
                continue
            article_text = extract(
                h,
                include_tables=False,
                include_comments=False,
                output_format="txt",
            )
            if article_text is not None and len(article_text) > self.min_char_count:
                articles[u] = {"text": article_text}

        return articles

    def urls_to_snippets(self, urls: List[str]) -> Dict:
        articles = self.urls_to_articles(urls)
        for u in articles:
            articles[u]["snippets"] = self.text_splitter.split_text(articles[u]["text"])

        return articles
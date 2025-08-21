"""Load html from files, clean up, split, ingest into Weaviate."""

import logging
import os
import re
from parser import langchain_docs_extractor

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from constants import WEAVIATE_DOCS_INDEX_NAME
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import Weaviate
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

# --- Logger setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Embeddings model ---
def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


# --- Metadata extraction ---
def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


# --- Load docs functions ---
def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


# --- Main ingestion function ---
def ingest_docs():
    # --- Load environment variables safely ---
    WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
    RECORD_MANAGER_DB_URL = os.environ.get("RECORD_MANAGER_DB_URL")

    if not WEAVIATE_URL or not WEAVIATE_API_KEY or not RECORD_MANAGER_DB_URL:
        raise ValueError(
            "Missing required environment variables: WEAVIATE_URL, WEAVIATE_API_KEY, or RECORD_MANAGER_DB_URL"
        )

    # --- Setup text splitter and embeddings ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    # --- Weaviate client & vectorstore ---
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
        additional_headers={"X-OpenAI-Project": "LangChain-Ingest"},  # optional safe header
    )

    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "title"],
    )

    # --- SQL record manager with NullPool to avoid EOF issues ---
    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    # --- Load all docs ---
    docs_from_documentation = load_langchain_docs()
    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")

    docs_from_api = load_api_docs()
    logger.info(f"Loaded {len(docs_from_api)} docs from API")

    docs_from_langsmith = load_langsmith_docs()
    logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")

    # --- Split & clean documents ---
    docs_transformed = text_splitter.split_documents(
        docs_from_documentation + docs_from_api + docs_from_langsmith
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # Ensure metadata safety
    for doc in docs_transformed:
        doc.metadata.setdefault("source", "")
        doc.metadata.setdefault("title", "")

    # --- Index into Weaviate ---
    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",  # can use "incremental" for faster runs
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )
    logger.info(f"Indexing stats: {indexing_stats}")

    # --- Optional: check vector count ---
    try:
        num_vecs = client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
        logger.info(f"LangChain now has this many vectors: {num_vecs}")
    except Exception as e:
        logger.warning(f"Could not fetch vector count: {e}")


if __name__ == "__main__":
    ingest_docs()

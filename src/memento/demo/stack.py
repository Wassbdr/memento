"""Connect the memory engine to the real Neo4j + ChromaDB + Ollama stack."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import platform
import subprocess
import time
from urllib import request

from memento.conversation import (
    DEFAULT_CONVERSATION_SYSTEM_INSTRUCTION,
    ConversationConfig,
    ConversationOrchestrator,
    OpenAICompatibleBackendConfig,
    OpenAICompatibleConversationBackend,
)
from memento.memory import ChromaSemanticIndex, MemorySyncEngine, Neo4jGraphStore

DEFAULT_DEMO_LLM_MODEL = "qwen2.5:0.5b"
DEMO_ANSWER_STYLE = " Reponds en une ou deux phrases, en vouvoyant la personne."
DEFAULT_DEMO_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


@dataclass(frozen=True)
class StackConfig:
    """Connection settings for the docker compose stack (overridable by env vars)."""

    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "memento-demo"))
    chroma_host: str = field(default_factory=lambda: _env("CHROMA_HOST", "localhost"))
    chroma_port: int = field(default_factory=lambda: int(_env("CHROMA_PORT", "8001")))
    chroma_collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "memento-demo"))
    embedding_model: str = field(default_factory=lambda: _env("MEMENTO_EMBEDDING_MODEL", DEFAULT_DEMO_EMBEDDING_MODEL))
    llm_base_url: str = field(default_factory=lambda: _env("MEMENTO_LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_model: str = field(default_factory=lambda: _env("MEMENTO_LLM_MODEL", DEFAULT_DEMO_LLM_MODEL))
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("MEMENTO_LLM_API_KEY") or None)


def build_store_engine(config: StackConfig | None = None) -> MemorySyncEngine:
    """Return a sync engine backed by Neo4j (graph) and ChromaDB (vectors)."""

    config = config or StackConfig()
    graph_store = Neo4jGraphStore(
        uri=config.neo4j_uri,
        username=config.neo4j_user,
        password=config.neo4j_password,
    )
    semantic_index = ChromaSemanticIndex(
        collection_name=config.chroma_collection,
        embedding_model=config.embedding_model,
        host=config.chroma_host,
        port=config.chroma_port,
    )
    return MemorySyncEngine(graph_store=graph_store, semantic_index=semantic_index)


def build_orchestrator(engine: MemorySyncEngine, config: StackConfig | None = None) -> ConversationOrchestrator:
    """Return the conversation orchestrator wired to the local OpenAI-compatible LLM."""

    config = config or StackConfig()
    backend = OpenAICompatibleConversationBackend(
        OpenAICompatibleBackendConfig(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            timeout_seconds=300.0,
        )
    )
    return ConversationOrchestrator(
        memory_engine=engine,
        backend=backend,
        config=ConversationConfig(
            model_name=config.llm_model,
            temperature=0.0,
            system_instruction=DEFAULT_CONVERSATION_SYSTEM_INSTRUCTION + DEMO_ANSWER_STYLE,
        ),
    )


def wait_for_stack(config: StackConfig | None = None, *, timeout_seconds: float = 180.0) -> None:
    """Block until Neo4j, ChromaDB and the LLM endpoint all answer."""

    config = config or StackConfig()
    deadline = time.monotonic() + timeout_seconds
    checks = {
        "neo4j": _neo4j_ready,
        "chroma": lambda c: _http_ok(f"http://{c.chroma_host}:{c.chroma_port}/api/v2/heartbeat"),
        "llm": lambda c: _http_ok(c.llm_base_url.rstrip("/") + "/models"),
    }
    pending = dict(checks)
    while pending:
        for name, check in list(pending.items()):
            if check(config):
                pending.pop(name)
        if not pending:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"stack services not ready: {sorted(pending)}")
        time.sleep(1.0)


def restart_stack(compose_dir: str, *, services: tuple[str, ...] = ("neo4j", "chroma", "ollama")) -> None:
    """Restart the docker compose services; named volumes (the stored memory) are kept."""

    subprocess.run(["docker", "compose", "restart", *services], cwd=compose_dir, check=True, capture_output=True)


def hardware_summary() -> str:
    """One-line description of the CPU this run executed on."""

    model = platform.processor() or "unknown CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    memory_gb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            memory_gb = round(int(handle.readline().split()[1]) / 1024 / 1024)
    except (OSError, ValueError, IndexError):
        pass
    memory = f", {memory_gb} GB RAM" if memory_gb else ""
    return f"{model}, {os.cpu_count()} vCPU{memory}, no GPU"


def _neo4j_ready(config: StackConfig) -> bool:
    try:
        from neo4j import GraphDatabase

        with GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)) as driver:
            driver.verify_connectivity()
        return True
    except Exception:
        return False


def _http_ok(url: str) -> bool:
    try:
        with request.urlopen(url, timeout=3) as response:
            return 200 <= response.status < 300
    except Exception:
        return False

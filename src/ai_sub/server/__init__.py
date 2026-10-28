"""Server coordinator module for distributed AI Sub execution."""

from ai_sub.server.app import create_app
from ai_sub.server.channel_mapping import ChannelInfo, ChannelMap, ChannelMapper
from ai_sub.server.config import (
    FilesSourceConfig,
    GitConfig,
    PipelineConfig,
    ServerOptions,
    ServerSettings,
    SourcesConfig,
    YoutubeSourceConfig,
)
from ai_sub.server.coordinator import PipelineCoordinator
from ai_sub.server.git_sync import GitSyncManager
from ai_sub.server.ingestion import DiscoveredVideo, IngestionManager, ensure_pot_provider
from ai_sub.server.models import (
    ActiveVideo,
    ActivityEvent,
    ServerTask,
    TaskStatus,
    generate_w3c_traceparent,
)
from ai_sub.server.pipeline import ComboPipeline

__all__ = [
    "ActiveVideo",
    "ActivityEvent",
    "ChannelInfo",
    "ChannelMap",
    "ChannelMapper",
    "ComboPipeline",
    "DiscoveredVideo",
    "FilesSourceConfig",
    "GitConfig",
    "GitSyncManager",
    "IngestionManager",
    "PipelineConfig",
    "PipelineCoordinator",
    "ServerOptions",
    "ServerSettings",
    "ServerTask",
    "SourcesConfig",
    "TaskStatus",
    "YoutubeSourceConfig",
    "create_app",
    "ensure_pot_provider",
    "generate_w3c_traceparent",
]

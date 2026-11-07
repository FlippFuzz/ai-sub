"""Server coordinator module for distributed AI Sub execution."""

from ai_sub.server.app import create_app
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
from ai_sub.server.ingestion import IngestionManager, QueuedVideo
from ai_sub.server.models import (
    ActiveVideo,
    ActivityEvent,
    ServerTask,
    TaskStatus,
)
from ai_sub.server.pipeline import ComboPipeline, PipelineStatus
from ai_sub.server.youtube import (
    DiscoveredVideo,
    DownloadedVideo,
    YouTubeBotDetectionError,
    YouTubeClient,
    YouTubeError,
    YouTubeVideoUnavailableError,
)

__all__ = [
    "ActiveVideo",
    "ActivityEvent",
    "ComboPipeline",
    "DiscoveredVideo",
    "DownloadedVideo",
    "FilesSourceConfig",
    "GitConfig",
    "GitSyncManager",
    "IngestionManager",
    "PipelineConfig",
    "PipelineCoordinator",
    "PipelineStatus",
    "QueuedVideo",
    "ServerOptions",
    "ServerSettings",
    "ServerTask",
    "SourcesConfig",
    "TaskStatus",
    "YouTubeBotDetectionError",
    "YouTubeClient",
    "YouTubeError",
    "YouTubeVideoUnavailableError",
    "YoutubeSourceConfig",
    "create_app",
]

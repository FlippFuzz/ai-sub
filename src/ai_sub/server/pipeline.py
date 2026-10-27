"""Pipeline coordinator managing 1 active video lifecycle, task creation, and state resumption."""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import logfire
from pysubs2 import SSAEvent, SSAFile

from ai_sub.config import ReEncodeSettings, SearchSettings, SplittingSettings
from ai_sub.jobs.lyrics import (
    LYRICS_PROMPT_VERSION,
    LyricsAiResponse,
    LyricsJob,
    get_lyrics_prompt,
)
from ai_sub.jobs.subtitles import (
    SUBTITLES_PROMPT_VERSION,
    DetectedMusicalPerformance,
    Subtitles,
    SubtitlesAiResponse,
    SubtitlesJob,
    get_subtitles_prompt,
)
from ai_sub.jobs.summary import (
    SUMMARY_PROMPT_VERSION,
    SummaryAiResponse,
    SummaryJob,
    get_summary_prompt,
)
from ai_sub.protocol import TaskType
from ai_sub.server.config import PipelineConfig
from ai_sub.server.models import ActiveVideo, ServerTask, TaskStatus
from ai_sub.shortcode import (
    generate_full_shortcode,
    generate_lyrics_shortcode,
    generate_subtitles_shortcode,
    generate_summary_shortcode,
)
from ai_sub.telemetry import get_app_version
from ai_sub.video import (
    get_video_duration_ms,
    reencode_video,
    split_video,
)


class ComboPipeline:
    """Manages an independent combination pipeline restricted to 1 active video."""

    config: PipelineConfig
    base_work_dir: Path
    work_dir: Path
    validation_buffer_ms: int
    reencode_settings: ReEncodeSettings
    search_settings: SearchSettings
    split_settings: SplittingSettings

    summary_shortcode: str
    subtitles_shortcode: str
    lyrics_shortcode: str
    full_shortcode: str

    active_video: ActiveVideo | None
    tasks: dict[str, ServerTask]
    summary_response: SummaryAiResponse | None
    _reencode_task: asyncio.Task[None] | None

    def __init__(
        self,
        config: PipelineConfig,
        base_work_dir: Path,
        reencode_settings: ReEncodeSettings,
        search_settings: SearchSettings,
        split_settings: SplittingSettings,
        validation_buffer_ms: int = 2000,
    ) -> None:
        """Initializes the combination pipeline runner.

        Args:
            config: Model pipeline configuration.
            base_work_dir: Root working directory containing per-pipeline folders.
            reencode_settings: Video compression and proxy rules.
            search_settings: Search tool configuration.
            split_settings: Video chunk duration and tolerance rules.
            validation_buffer_ms: Allowed timestamp tolerance in milliseconds. Defaults to 2000.
        """
        self.config = config
        self.base_work_dir = base_work_dir
        self.work_dir = base_work_dir / config.name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.validation_buffer_ms = validation_buffer_ms

        self.reencode_settings = reencode_settings
        self.search_settings = search_settings
        self.split_settings = split_settings

        self.summary_shortcode = generate_summary_shortcode(config.summary_model)
        self.subtitles_shortcode = generate_subtitles_shortcode(
            model_name=config.subtitles_model,
            summary_shortcode=self.summary_shortcode,
        )
        self.lyrics_shortcode = generate_lyrics_shortcode(
            lyrics_model_name=config.lyrics_model,
            subtitles_shortcode=self.subtitles_shortcode,
        )
        self.full_shortcode = generate_full_shortcode(config.subtitles_model)

        self.active_video = None
        self.tasks = {}
        self.summary_response = None
        self._reencode_task = None

    def is_idle(self) -> bool:
        """Checks if the pipeline is idle and ready to accept a new video.

        Returns:
            bool: True if no video is currently active; False otherwise.
        """
        return self.active_video is None

    def _build_user_context(self) -> str | None:
        """Constructs background user context from video metadata.

        Returns:
            str | None: Formatted context string or None.
        """
        if not self.active_video:
            return None
        if self.active_video.source_type == "youtube":
            ch = self.active_video.channel_name or self.active_video.uploader
            return f"This is a YouTube video from {ch} channel. The video title is '{self.active_video.title}'."
        return f"The video title is '{self.active_video.title}'."

    def _get_active_search_tools(self) -> list[str]:
        """Resolves active search tools from pipeline override or global search settings.

        Returns:
            list[str]: Active search tools list.
        """
        if self.config.search_tools is not None:
            return self.config.search_tools
        return self.search_settings.tools

    async def _reencode_segments_background(self) -> None:
        """Background loop re-encoding segments sequentially using the serial FFmpeg lock."""
        if not self.active_video:
            return

        active = self.active_video
        reencode_dir = active.job_dir / "reencoded"
        reencode_dir.mkdir(parents=True, exist_ok=True)

        seg_cfg = self.reencode_settings.segment
        with logfire.span(
            "Background Re-encoding Loop: {video_id}",
            combo=self.config.name,
            video_id=active.video_id,
            enabled=seg_cfg.enabled,
            threshold_mb=seg_cfg.threshold_mb,
            total_chunks=len(active.segment_paths),
            _level="debug",
        ) as loop_span:
            reencoded_count = 0
            skipped_count = 0

            for idx, seg_path in enumerate(active.segment_paths):
                file_size_mb = seg_path.stat().st_size / (1024 * 1024)
                should_reencode = False
                skip_reason = None

                if not seg_cfg.enabled:
                    skip_reason = "reencode.segment.enabled is False"
                elif seg_cfg.threshold_mb > 0 and file_size_mb < seg_cfg.threshold_mb:
                    skip_reason = f"size ({file_size_mb:.1f} MB) < threshold ({seg_cfg.threshold_mb} MB)"
                else:
                    should_reencode = True

                if not should_reencode:
                    active.reencoded_segment_paths[idx] = seg_path
                    skipped_count += 1
                    logfire.debug(
                        f"[{self.config.name}] Chunk {seg_path.name} ({file_size_mb:.1f} MB) "
                        f"re-encoding skipped ({skip_reason})."
                    )
                    continue

                out_file = reencode_dir / f"{seg_path.stem}.mp4"
                if out_file.is_file():
                    active.reencoded_segment_paths[idx] = out_file
                    reencoded_count += 1
                    continue

                with logfire.span(
                    "Re-encoding Chunk: {chunk}",
                    chunk=seg_path.name,
                    combo=self.config.name,
                    video_id=active.video_id,
                    size_mb=round(file_size_mb, 2),
                    _level="debug",
                ) as chunk_span:
                    try:
                        await reencode_video(
                            input_path=seg_path,
                            output_path=out_file,
                            fps=seg_cfg.fps,
                            height=seg_cfg.height,
                            bitrate_kb=seg_cfg.bitrate_kb,
                            encoder=self.reencode_settings.encoder or "libx264",
                            duration_tolerance_ms=seg_cfg.duration_tolerance_ms,
                            keyframe_only=seg_cfg.keyframe_only,
                        )
                        active.reencoded_segment_paths[idx] = out_file
                        reencoded_count += 1
                        out_size_mb = out_file.stat().st_size / (1024 * 1024)
                        chunk_span.set_attribute("result", "reencoded")
                        chunk_span.set_attribute("output_size_mb", round(out_size_mb, 2))
                        logfire.info(
                            f"[{self.config.name}] Chunk {seg_path.name} ({file_size_mb:.1f} MB) "
                            f"re-encoded to {out_file.name} ({out_size_mb:.1f} MB)."
                        )

                        task_id = f"{self.config.name}::{active.video_id}::subtitles::{idx:03d}"
                        if task_id in self.tasks:
                            self.tasks[task_id].media_path = out_file
                    except Exception as e:
                        chunk_span.set_attribute("result", "failed")
                        chunk_span.set_attribute("error", str(e))
                        logfire.warning(f"Re-encode failed for chunk {seg_path.name}: {e}. Falling back to raw chunk.")
                        active.reencoded_segment_paths[idx] = seg_path

            loop_span.set_attribute("reencoded_chunks", reencoded_count)
            loop_span.set_attribute("skipped_chunks", skipped_count)
            logfire.info(
                f"[{self.config.name}] Background re-encoding finished for {active.video_id}: "
                f"{reencoded_count} re-encoded, {skipped_count} kept in native format."
            )

    async def prepare_and_set_video(
        self,
        video_id: str,
        uploader: str,
        title: str,
        raw_video_path: Path,
        channel_name: str | None = None,
        source_type: Literal["local", "youtube"] = "local",
    ) -> None:
        """Prepares a video inside work/<combo>/<video_id>/, saving metadata and splitting chunks.

        Args:
            video_id: Unique video identifier.
            uploader: Channel or uploader handle identifier.
            title: Title of the video.
            raw_video_path: Path to the local raw video file in this combo directory.
            channel_name: Human-readable channel display name. Defaults to None.
            source_type: Ingestion source type ('local' or 'youtube'). Defaults to 'local'.
        """
        job_dir = self.work_dir / video_id
        job_dir.mkdir(parents=True, exist_ok=True)

        with logfire.span(
            "Prepare Video: {video_id}",
            combo=self.config.name,
            video_id=video_id,
            title=title,
            uploader=uploader,
        ) as prep_span:
            total_duration_ms = await get_video_duration_ms(raw_video_path)
            raw_size_mb = raw_video_path.stat().st_size / (1024 * 1024)

            meta_file = job_dir / "video_meta.json"
            meta_data = {
                "video_id": video_id,
                "uploader": uploader,
                "channel_name": channel_name,
                "title": title,
                "source_type": source_type,
                "duration_ms": total_duration_ms,
                "raw_size_mb": round(raw_size_mb, 2),
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)

            sum_cfg = self.reencode_settings.summary
            should_reencode_summary = False
            summary_skip_reason = None

            if not sum_cfg.enabled:
                summary_skip_reason = "reencode.summary.enabled is False"
            elif sum_cfg.threshold_mb > 0 and raw_size_mb < sum_cfg.threshold_mb:
                summary_skip_reason = f"size ({raw_size_mb:.1f} MB) < threshold ({sum_cfg.threshold_mb} MB)"
            else:
                should_reencode_summary = True

            with logfire.span(
                "Check Summary Proxy: {video_id}",
                combo=self.config.name,
                video_id=video_id,
                raw_size_mb=round(raw_size_mb, 2),
                threshold_mb=sum_cfg.threshold_mb,
                enabled=sum_cfg.enabled,
                _level="debug",
            ) as proxy_span:
                if should_reencode_summary:
                    proxy_path = job_dir / "summary_proxy.mp4"
                    logfire.info(
                        f"[{self.config.name}] Compressing summary proxy ({raw_size_mb:.1f} MB >= "
                        f"{sum_cfg.threshold_mb} MB threshold)..."
                    )
                    await reencode_video(
                        input_path=raw_video_path,
                        output_path=proxy_path,
                        fps=sum_cfg.fps,
                        height=sum_cfg.height,
                        bitrate_kb=sum_cfg.bitrate_kb,
                        encoder=self.reencode_settings.encoder or "libx264",
                        duration_tolerance_ms=sum_cfg.duration_tolerance_ms,
                        keyframe_only=sum_cfg.keyframe_only,
                    )
                    proxy_size_mb = proxy_path.stat().st_size / (1024 * 1024)
                    proxy_span.set_attribute("result", "reencoded")
                    proxy_span.set_attribute("proxy_size_mb", round(proxy_size_mb, 2))
                    logfire.info(
                        f"[{self.config.name}] Summary proxy generated: {proxy_path.name} ({proxy_size_mb:.1f} MB)."
                    )
                else:
                    proxy_path = raw_video_path
                    proxy_span.set_attribute("result", "skipped")
                    proxy_span.set_attribute("skip_reason", summary_skip_reason)
                    logfire.info(
                        f"[{self.config.name}] Summary proxy re-encoding skipped: {summary_skip_reason}. "
                        "Using raw video directly."
                    )

            with logfire.span(
                "Split Video into Segments: {video_id}",
                combo=self.config.name,
                video_id=video_id,
                split_duration_s=self.split_settings.max_seconds,
            ) as split_span:
                segment_paths = await split_video(
                    input_video=raw_video_path,
                    output_dir=job_dir,
                    split_duration_s=self.split_settings.max_seconds,
                    output_pattern="part_%03d",
                    duration_tolerance_ms=self.split_settings.duration_tolerance_ms,
                )
                durations = await asyncio.gather(*(get_video_duration_ms(p) for p in segment_paths))
                split_span.set_attribute("chunks_created", len(segment_paths))
                logfire.info(
                    f"[{self.config.name}] Split '{video_id}' into {len(segment_paths)} chunks "
                    f"(probed total {sum(durations)}ms)."
                )

            self.active_video = ActiveVideo(
                video_id=video_id,
                uploader=uploader,
                channel_name=channel_name,
                title=title,
                source_type=source_type,
                job_dir=job_dir,
                raw_path=raw_video_path,
                duration_ms=total_duration_ms,
                summary_proxy_path=proxy_path,
                segment_paths=segment_paths,
                segment_durations_ms=durations,
            )
            self.active_video.user_context = self._build_user_context()
            self.tasks.clear()
            self.summary_response = None

            self._queue_summary_task()
            self._reencode_task = asyncio.create_task(self._reencode_segments_background())

            prep_span.set_attribute("result", "ready")
            prep_span.set_attribute("chunk_count", len(segment_paths))

    def _queue_summary_task(self) -> None:
        """Generates and enqueues the Phase 1 whole-video summary task."""
        if not self.active_video or not self.active_video.summary_proxy_path:
            return

        active = self.active_video
        proxy_path = active.summary_proxy_path
        assert proxy_path is not None

        task_id = f"{self.config.name}::{active.video_id}::summary"
        prompt = get_summary_prompt(user_context=active.user_context)

        active_tools = [t for t in self._get_active_search_tools() if t != "lrclib"]

        task = ServerTask(
            task_id=task_id,
            task_type=TaskType.SUMMARY,
            pipeline_name=self.config.name,
            video_id=active.video_id,
            model=self.config.summary_model,
            media_path=proxy_path,
            video_duration_ms=active.duration_ms,
            prompt=prompt,
            response_schema_name="SummaryAiResponse",
            search_tools=active_tools,
            agentic_video=True,
        )
        self.tasks[task_id] = task
        logfire.info(
            f"[{self.config.name}] Queued summary task for {active.video_id} "
            f"(model: {self.config.summary_model}, media: {proxy_path.name})."
        )

    def _queue_subtitles_tasks(self) -> None:
        """Generates and enqueues Phase 2 subtitle chunk tasks for all segments."""
        if not self.active_video:
            return

        active = self.active_video
        with logfire.span(
            "Queue Subtitles Tasks: {video_id}",
            combo=self.config.name,
            video_id=active.video_id,
            chunk_count=len(active.segment_paths),
            _level="debug",
        ):
            current_offset_ms = 0
            for idx, (seg_path, seg_dur_ms) in enumerate(
                zip(active.segment_paths, active.segment_durations_ms, strict=True)
            ):
                task_id = f"{self.config.name}::{active.video_id}::subtitles::{idx:03d}"
                prompt = get_subtitles_prompt(
                    user_context=active.user_context,
                    global_summary=self.summary_response,
                    segment_offset_ms=current_offset_ms,
                    segment_duration_ms=seg_dur_ms,
                )

                media_file = active.reencoded_segment_paths.get(idx) or seg_path

                task = ServerTask(
                    task_id=task_id,
                    task_type=TaskType.SUBTITLES,
                    pipeline_name=self.config.name,
                    video_id=active.video_id,
                    segment_index=idx,
                    model=self.config.subtitles_model,
                    media_path=media_file,
                    video_duration_ms=seg_dur_ms,
                    prompt=prompt,
                    response_schema_name="SubtitlesAiResponse",
                    search_tools=[],
                    agentic_video=False,
                )
                self.tasks[task_id] = task
                current_offset_ms += seg_dur_ms

            logfire.info(
                f"[{self.config.name}] Queued {len(active.segment_paths)} subtitle tasks "
                f"for {active.video_id} (model: {self.config.subtitles_model})."
            )

    def _queue_lyrics_task(self, seg_idx: int, sub_resp: SubtitlesAiResponse) -> None:
        """Generates and enqueues a Phase 3 lyrics task for a segment with detected singing.

        Args:
            seg_idx: Segment chunk index.
            sub_resp: Subtitle response containing detected singing performances.
        """
        if not self.active_video:
            return

        active = self.active_video
        seg_path = active.reencoded_segment_paths.get(seg_idx) or active.segment_paths[seg_idx]
        seg_dur_ms = active.segment_durations_ms[seg_idx]
        offset_ms = sum(active.segment_durations_ms[:seg_idx])

        task_id = f"{self.config.name}::{active.video_id}::lyrics::{seg_idx:03d}"
        prompt = get_lyrics_prompt(
            draft_subtitles=sub_resp.subtitles,
            detected_performances=sub_resp.detected_musical_performances,
            user_context=active.user_context,
            global_summary=self.summary_response,
            segment_offset_ms=offset_ms,
            segment_duration_ms=seg_dur_ms,
        )

        active_tools = self._get_active_search_tools()

        task = ServerTask(
            task_id=task_id,
            task_type=TaskType.LYRICS,
            pipeline_name=self.config.name,
            video_id=active.video_id,
            segment_index=seg_idx,
            model=self.config.lyrics_model,
            media_path=seg_path,
            video_duration_ms=seg_dur_ms,
            prompt=prompt,
            response_schema_name="LyricsAiResponse",
            search_tools=active_tools,
            agentic_video=True,
            detected_performances=[p.model_dump(mode="json") for p in sub_resp.detected_musical_performances],
            draft_subtitles=[s.model_dump(mode="json") for s in sub_resp.subtitles],
        )
        self.tasks[task_id] = task
        logfire.info(
            f"[{self.config.name}] Queued lyrics task for segment {seg_idx:03d} of {active.video_id} "
            f"({len(sub_resp.detected_musical_performances)} musical performances detected)."
        )

    def on_task_completed(self, task_id: str, result_data: dict[str, Any]) -> None:
        """Handles task completion, saving YAML state files and triggering downstream stages.

        Args:
            task_id: Unique task identifier.
            result_data: Validated structured dictionary received from worker.
        """
        task = self.tasks.get(task_id)
        if not task or not self.active_video:
            return

        active = self.active_video

        with logfire.span(
            "Task Cascade Complete: {task_id}",
            combo=self.config.name,
            task_id=task_id,
            task_type=task.task_type.value,
        ) as cascade_span:
            task.status = TaskStatus.COMPLETED
            task.result = result_data

            if task.task_type == TaskType.SUMMARY:
                resp = SummaryAiResponse.model_validate(
                    result_data,
                    context={
                        "video_duration_ms": active.duration_ms,
                        "validation_buffer_ms": self.validation_buffer_ms,
                    },
                )
                self.summary_response = resp

                summary_job = SummaryJob(
                    name="summary",
                    video_duration_ms=active.duration_ms,
                    user_context=active.user_context,
                    response=resp,
                )
                summary_file = active.job_dir / f"summary.{self.summary_shortcode}.yaml"
                summary_job.save(summary_file)

                cascade_span.set_attribute("chapters_count", len(resp.chapters))
                cascade_span.set_attribute("entities_count", len(resp.entities))
                cascade_span.set_attribute("glossary_count", len(resp.domain_glossary))
                logfire.info(
                    f"[{self.config.name}] Summary task complete: extracted {len(resp.chapters)} chapters, "
                    f"{len(resp.entities)} entities, {len(resp.domain_glossary)} glossary terms. "
                    f"Saved to {summary_file.name}."
                )

                self._queue_subtitles_tasks()

            elif task.task_type == TaskType.SUBTITLES:
                assert task.segment_index is not None
                idx = task.segment_index
                seg_dur_ms = active.segment_durations_ms[idx]
                offset_ms = sum(active.segment_durations_ms[:idx])

                resp = SubtitlesAiResponse.model_validate(
                    result_data,
                    context={
                        "video_duration_ms": seg_dur_ms,
                        "validation_buffer_ms": self.validation_buffer_ms,
                    },
                )

                seg_path = active.segment_paths[idx]
                sub_job = SubtitlesJob(
                    name=seg_path.stem,
                    video_duration_ms=seg_dur_ms,
                    segment_offset_ms=offset_ms,
                    global_summary=self.summary_response,
                    user_context=active.user_context,
                    response=resp,
                )
                sub_file = active.job_dir / f"{seg_path.stem}.subtitles.{self.subtitles_shortcode}.yaml"
                sub_job.save(sub_file)

                cascade_span.set_attribute("subtitles_count", len(resp.subtitles))
                cascade_span.set_attribute("music_count", len(resp.detected_musical_performances))
                logfire.info(
                    f"[{self.config.name}] Subtitle chunk {idx:03d} complete: "
                    f"{len(resp.subtitles)} lines transcribed, "
                    f"{len(resp.detected_musical_performances)} singing performances detected. "
                    f"Saved to {sub_file.name}."
                )

                if resp.detected_musical_performances:
                    self._queue_lyrics_task(idx, resp)

            elif task.task_type == TaskType.LYRICS:
                assert task.segment_index is not None
                idx = task.segment_index
                seg_dur_ms = active.segment_durations_ms[idx]
                offset_ms = sum(active.segment_durations_ms[:idx])
                seg_path = active.segment_paths[idx]

                resp = LyricsAiResponse.model_validate(
                    result_data,
                    context={
                        "video_duration_ms": seg_dur_ms,
                        "validation_buffer_ms": self.validation_buffer_ms,
                    },
                )

                lyrics_job = LyricsJob(
                    name=seg_path.stem,
                    video_duration_ms=seg_dur_ms,
                    segment_offset_ms=offset_ms,
                    draft_subtitles=[Subtitles.model_validate(s) for s in task.draft_subtitles],
                    detected_performances=[
                        DetectedMusicalPerformance.model_validate(p) for p in task.detected_performances
                    ],
                    global_summary=self.summary_response,
                    user_context=active.user_context,
                    response=resp,
                )
                lyrics_file = active.job_dir / f"{seg_path.stem}.lyrics.{self.lyrics_shortcode}.yaml"
                lyrics_job.save(lyrics_file)

                cascade_span.set_attribute("refined_subtitles_count", len(resp.subtitles))
                logfire.info(
                    f"[{self.config.name}] Lyrics refinement for chunk {idx:03d} complete: "
                    f"{len(resp.subtitles)} lines polished and saved to {lyrics_file.name}."
                )

    def is_video_complete(self) -> bool:
        """Checks if all segments for the active video have finished processing.

        Returns:
            bool: True if all queued tasks are completed; False otherwise.
        """
        if not self.active_video or not self.tasks:
            return False

        active = self.active_video

        if any(t.status != TaskStatus.COMPLETED for t in self.tasks.values()):
            return False

        sub_tasks = [t for t in self.tasks.values() if t.task_type == TaskType.SUBTITLES]
        if len(sub_tasks) < len(active.segment_paths):
            return False

        return True

    def stitch_final_subtitles(self, output_path: Path) -> Path:
        """Stitches completed segment subtitles into a unified SRT file.

        Args:
            output_path: Destination path for the final stitched SRT file.

        Returns:
            Path: The generated subtitle file path.

        Raises:
            RuntimeError: If video processing is incomplete or segments are missing.
        """
        if not self.active_video or not self.is_video_complete():
            raise RuntimeError(f"Cannot stitch subtitles: {self.config.name} video processing is incomplete.")

        active = self.active_video

        with logfire.span(
            "Stitch Subtitles: {video_id}",
            combo=self.config.name,
            video_id=active.video_id,
            output_file=output_path.name,
        ) as stitch_span:
            all_subs = SSAFile()
            current_offset_ms = 0

            for idx, (seg_path, seg_dur_ms) in enumerate(
                zip(active.segment_paths, active.segment_durations_ms, strict=True)
            ):
                lyrics_file = active.job_dir / f"{seg_path.stem}.lyrics.{self.lyrics_shortcode}.yaml"
                sub_file = active.job_dir / f"{seg_path.stem}.subtitles.{self.subtitles_shortcode}.yaml"

                active_resp = None
                if lyrics_file.is_file():
                    lyr_job = LyricsJob.load(lyrics_file, self.validation_buffer_ms)
                    if lyr_job and lyr_job.response:
                        active_resp = lyr_job.response
                elif sub_file.is_file():
                    s_job = SubtitlesJob.load(sub_file, self.validation_buffer_ms)
                    if s_job and s_job.response:
                        active_resp = s_job.response

            if active_resp:
                seg_subs = active_resp.get_ssafile()
                seg_subs.shift(ms=current_offset_ms)
                all_subs += seg_subs
            else:
                all_subs.append(
                    SSAEvent(
                        start=current_offset_ms,
                        end=current_offset_ms + seg_dur_ms,
                        text="Error processing subtitles for this segment.",
                    )
                )

            current_offset_ms += seg_dur_ms

            metadata = {
                "ai_sub_version": get_app_version(),
                "pipeline": self.config.name,
                "generated_at": datetime.now().isoformat(),
                "summary_prompt_version": SUMMARY_PROMPT_VERSION,
                "lyrics_prompt_version": LYRICS_PROMPT_VERSION,
                "subtitles_prompt_version": SUBTITLES_PROMPT_VERSION,
                "complete": True,
            }
            info_str = json.dumps(metadata, indent=2).replace("{", "(").replace("}", ")")
            all_subs.insert(0, SSAEvent(start=0, end=1, text=info_str))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            all_subs.save(str(output_path))

            stitch_span.set_attribute("total_events", len(all_subs))
            stitch_span.set_attribute("duration_ms", current_offset_ms)
            logfire.info(
                f"[{self.config.name}] Stitched {len(all_subs)} subtitle events across "
                f"{len(active.segment_paths)} chunks to {output_path.name}."
            )
            return output_path

    async def resume_from_disk(self) -> bool:
        """Inspects work/<combo>/ to restore in-progress video subdirectories upon server boot.

        Returns:
            bool: True if an active video was successfully restored; False otherwise.
        """
        valid_extensions = {".mp4", ".mkv", ".webm", ".mov", ".flv", ".ts"}

        with logfire.span("Disk Resume Check: {combo}", combo=self.config.name) as resume_span:
            job_dirs = [d for d in self.work_dir.iterdir() if d.is_dir() and d.name != "reencoded"]
            if not job_dirs:
                resume_span.set_attribute("result", "idle")
                logfire.debug(f"[{self.config.name}] No active job subdirectories found. Pipeline is idle.")
                return False

            job_dir = job_dirs[0]
            meta_file = job_dir / "video_meta.json"
            if not meta_file.is_file():
                logfire.warning(f"[{self.config.name}] Job folder '{job_dir.name}' lacks video_meta.json. Cleaning.")
                shutil.rmtree(job_dir, ignore_errors=True)
                resume_span.set_attribute("result", "purged_invalid_folder")
                return False

            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                video_id = str(meta.get("video_id", job_dir.name))
                uploader = str(meta.get("uploader", "unknown"))
                channel_name = meta.get("channel_name")
                title = str(meta.get("title", job_dir.name))
                source_type = meta.get("source_type", "youtube")
            except Exception as e:
                logfire.warning(f"Could not load {meta_file.name}: {e}. Skipping resume.")
                resume_span.set_attribute("result", "corrupt_metadata")
                return False

            segment_files = sorted(
                [p for p in job_dir.glob("part_*") if p.is_file() and p.suffix.lower() in valid_extensions]
            )
            if not segment_files:
                logfire.warning(f"[{self.config.name}] Job '{video_id}' has no split chunks. Cleaning.")
                shutil.rmtree(job_dir, ignore_errors=True)
                resume_span.set_attribute("result", "no_chunks")
                return False

            proxy_path = job_dir / "summary_proxy.mp4"
            if not proxy_path.is_file():
                raw_candidates = [p for p in job_dir.glob(f"{video_id}.*") if p.is_file()]
                proxy_path = raw_candidates[0] if raw_candidates else segment_files[0]

            durations = await asyncio.gather(*(get_video_duration_ms(p) for p in segment_files))
            total_duration_ms = sum(durations)

            reencode_dir = job_dir / "reencoded"
            reencoded_paths = {}
            for idx, seg_path in enumerate(segment_files):
                reenc_file = reencode_dir / f"{seg_path.stem}.mp4"
                if reenc_file.is_file():
                    reencoded_paths[idx] = reenc_file

            self.active_video = ActiveVideo(
                video_id=video_id,
                uploader=uploader,
                channel_name=channel_name,
                title=title,
                source_type=source_type,
                job_dir=job_dir,
                raw_path=proxy_path,
                duration_ms=total_duration_ms,
                summary_proxy_path=proxy_path,
                segment_paths=segment_files,
                reencoded_segment_paths=reencoded_paths,
                segment_durations_ms=durations,
            )
            self.active_video.user_context = self._build_user_context()
            active = self.active_video
            self.tasks.clear()

            # Always start background chunk re-encoding on resume
            self._reencode_task = asyncio.create_task(self._reencode_segments_background())

            summary_file = job_dir / f"summary.{self.summary_shortcode}.yaml"
            loaded_summary_job = SummaryJob.load(summary_file, self.validation_buffer_ms)

            if not loaded_summary_job or not loaded_summary_job.response:
                logfire.info(
                    f"[{self.config.name}] Resuming: summary incomplete for '{video_id}'. Queueing summary task."
                )
                self._queue_summary_task()
                resume_span.set_attribute("result", "resumed_summary_pending")
                return True

            self.summary_response = loaded_summary_job.response
            logfire.info(f"[{self.config.name}] Resuming: loaded valid summary for '{video_id}'.")

            current_offset_ms = 0
            for idx, (seg_path, seg_dur_ms) in enumerate(zip(segment_files, durations, strict=True)):
                sub_file = job_dir / f"{seg_path.stem}.subtitles.{self.subtitles_shortcode}.yaml"
                lyrics_file = job_dir / f"{seg_path.stem}.lyrics.{self.lyrics_shortcode}.yaml"

                loaded_sub_job = SubtitlesJob.load(sub_file, self.validation_buffer_ms)
                loaded_lyrics_job = LyricsJob.load(lyrics_file, self.validation_buffer_ms)

                media_file = reencoded_paths.get(idx) or seg_path

                if not loaded_sub_job or not loaded_sub_job.response:
                    task_id = f"{self.config.name}::{video_id}::subtitles::{idx:03d}"
                    prompt = get_subtitles_prompt(
                        user_context=active.user_context,
                        global_summary=self.summary_response,
                        segment_offset_ms=current_offset_ms,
                        segment_duration_ms=seg_dur_ms,
                    )
                    self.tasks[task_id] = ServerTask(
                        task_id=task_id,
                        task_type=TaskType.SUBTITLES,
                        pipeline_name=self.config.name,
                        video_id=video_id,
                        segment_index=idx,
                        model=self.config.subtitles_model,
                        media_path=media_file,
                        video_duration_ms=seg_dur_ms,
                        prompt=prompt,
                        response_schema_name="SubtitlesAiResponse",
                    )
                elif loaded_sub_job.response.detected_musical_performances:
                    if not loaded_lyrics_job or not loaded_lyrics_job.response:
                        self._queue_lyrics_task(idx, loaded_sub_job.response)

                current_offset_ms += seg_dur_ms

            resume_span.set_attribute("result", "resumed_subtitles_pending")
            resume_span.set_attribute("pending_tasks", len(self.tasks))
            logfire.info(f"[{self.config.name}] Resumed '{video_id}' from disk with {len(self.tasks)} pending tasks.")
            return True

    def clear_active_video(self) -> None:
        """Purges working directory files and clears the active video state for this pipeline."""
        with logfire.span("Clear Active Video: {combo}", combo=self.config.name) as clear_span:
            if self._reencode_task and not self._reencode_task.done():
                self._reencode_task.cancel()
                self._reencode_task = None

            purged_path = None
            if self.active_video and self.active_video.job_dir.exists():
                purged_path = str(self.active_video.job_dir)
                try:
                    shutil.rmtree(self.active_video.job_dir)
                except OSError as e:
                    logfire.warning(f"Could not purge job directory {self.active_video.job_dir.name}: {e}")

            clear_span.set_attribute("purged_directory", purged_path or "none")
            self.active_video = None
            self.tasks.clear()
            self.summary_response = None
            logfire.info(f"[{self.config.name}] Job directory cleared. Pipeline is idle.")

"""Embedded live WebUI dashboard HTML template for ai-sub-server."""

from __future__ import annotations

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Sub Coordinator</title>
  <style>
    :root {
      --bg: #0f172a; --panel: #1e293b; --card-inner: #0f172a; --text: #f8fafc;
      --muted: #94a3b8; --border: #334155; --accent: #38bdf8;
      --green: #22c55e; --yellow: #eab308; --red: #ef4444;
      --blue: #38bdf8; --purple: #a855f7; --pink: #ec4899;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg); color: var(--text); margin: 0; padding: 24px;
    }
    h1, h2, h3, h4 { margin-top: 0; font-weight: 600; }
    .header {
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px;
    }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 20px; }
    .card {
      background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
      padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .stage-card {
      background: var(--card-inner); border: 1px solid var(--border); border-radius: 6px;
      padding: 16px; display: flex; flex-direction: column; justify-content: space-between; min-height: 110px;
    }

    /* Multi-segment Stacked Progress Bar */
    .progress-bar {
      background: #1e293b; border-radius: 9999px; height: 8px; overflow: hidden;
      margin: 8px 0; border: 1px solid var(--border); display: flex;
    }
    .progress-fill-done { background: var(--green); height: 100%; transition: width 0.3s; }
    .progress-fill-prog { background: var(--blue); height: 100%; transition: width 0.3s; }

    /* Standardized Badges */
    .badge {
      display: inline-block; padding: 3px 8px; border-radius: 4px;
      font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.025em;
    }
    .badge-claim, .badge-claimed, .badge-processing, .badge-in_progress { background: rgba(56, 189, 248, 0.2); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.4); }
    .badge-complete, .badge-completed { background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-fail, .badge-failed, .badge-push_failed { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-timeout, .badge-preparing { background: rgba(234, 179, 8, 0.2); color: #fde047; border: 1px solid rgba(234, 179, 8, 0.4); }
    .badge-pending, .badge-idle, .badge-skipped { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid var(--border); }
    .badge-finalizing { background: rgba(168, 85, 247, 0.2); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.4); }
    .badge-youtube { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-local { background: rgba(56, 189, 248, 0.15); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.3); }

    .btn {
      background: #334155; color: var(--text); border: 1px solid var(--border);
      padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500;
    }
    .btn:hover { background: #475569; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-weight: 500; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }

    /* Tabs */
    .tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
    .tab-btn {
      background: transparent; border: none; color: var(--muted); padding: 10px 18px;
      font-size: 14px; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent;
      display: inline-flex; align-items: center; gap: 8px;
    }
    .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

    /* Expandable activity details */
    .expand-btn {
      background: transparent; border: 1px solid var(--border); color: var(--muted);
      padding: 2px 8px; border-radius: 4px; font-size: 11px; cursor: pointer;
    }
    .expand-btn:hover { color: var(--text); border-color: var(--muted); }
    .detail-row { background: var(--card-inner); font-family: monospace; font-size: 12px; color: #cbd5e1; display: none; }
    .detail-box { padding: 12px 16px; white-space: pre-wrap; word-break: break-all; margin: 0; line-height: 1.5; }
    .detail-box-fail { background: rgba(239, 68, 68, 0.08); border-left: 3px solid var(--red); color: #fca5a5; }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>AI Sub Coordinator</h1>
      <span class="mono" style="color: var(--muted);" id="server-time">Loading...</span>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <button class="btn" onclick="updateDashboard()">⟳ Refresh UI</button>
      <span class="badge badge-complete">Live (auto-refresh 60s)</span>
    </div>
  </div>

  <div class="tabs" id="pipeline-tabs"></div>
  <div id="tab-content"></div>

  <div class="card" style="margin-top: 24px;">
    <h3>Recent Worker Activity</h3>
    <table id="activity-table">
      <thead><tr><th>Time</th><th>Worker</th><th>Pipeline</th><th>Task</th><th>Event</th><th>Action</th></tr></thead>
      <tbody><tr><td colspan="6" style="color: var(--muted);">Loading...</td></tr></tbody>
    </table>
  </div>

  <script>
    let currentData = null;
    let selectedPipelineIndex = 0;

    function selectTab(index) {
      selectedPipelineIndex = index;
      renderPipelineView();
    }

    function toggleDetail(idx) {
      const el = document.getElementById('detail-' + idx);
      if (el) {
        el.style.display = el.style.display === 'table-row' ? 'none' : 'table-row';
      }
    }

    function renderPipelineView() {
      if (!currentData || !currentData.pipelines) return;
      const pipe = currentData.pipelines[selectedPipelineIndex] || currentData.pipelines[0];
      if (!pipe) return;

      const tabs = document.getElementById('pipeline-tabs');
      tabs.innerHTML = currentData.pipelines.map((p, idx) => {
        const pStatus = p.status || (p.idle ? 'idle' : 'processing');
        return `
          <button class="tab-btn ${idx === selectedPipelineIndex ? 'active' : ''}" onclick="selectTab(${idx})">
            ${p.name}
            <span class="badge badge-${pStatus}">${pStatus}</span>
          </button>
        `;
      }).join('');

      const v = pipe.active_video;
      const st = pipe.stages;

      let activeSection = '';
      if (v && st) {
        const sum = st.summary;
        const reenc = st.reencoding;
        const sub = st.subtitles;
        const lyr = st.lyrics;

        // Exact Stacked Widths
        const sumDonePct = sum.done > 0 ? 100 : 0;
        const sumProgPct = sum.in_progress > 0 ? 100 : 0;

        const reencDonePct = reenc.total > 0 ? Math.round((reenc.done / reenc.total) * 100) : 100;
        const reencProgPct = reenc.total > 0 ? Math.round((reenc.in_progress / reenc.total) * 100) : 0;

        const subDonePct = sub.total > 0 ? Math.round((sub.done / sub.total) * 100) : 0;
        const subProgPct = sub.total > 0 ? Math.round((sub.in_progress / sub.total) * 100) : 0;

        const lyrDonePct = lyr.total > 0 ? Math.round((lyr.done / lyr.total) * 100) : 0;
        const lyrProgPct = lyr.total > 0 ? Math.round((lyr.in_progress / lyr.total) * 100) : 0;

        const durMin = Math.floor(v.duration_ms / 60000);
        const durSec = Math.floor((v.duration_ms % 60000) / 1000);
        const durationFormatted = durMin > 0 ? `${durMin}m ${durSec}s` : `${durSec}s`;

        activeSection = `
          <div class="card" style="margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 16px;">
              <div>
                <div style="display:flex; gap: 8px; align-items:center; margin-bottom: 6px;">
                  <span class="badge badge-complete">Active Video</span>
                  <span class="badge badge-${v.source_type}">${v.source_type}</span>
                  <span class="badge badge-${pipe.status || 'processing'}">${pipe.status || 'processing'}</span>
                </div>
                <h3 style="margin-bottom: 4px;">${v.title}</h3>
                <div style="color: var(--muted); font-size: 13px;">
                  Channel: <b>${v.channel_name || v.uploader}</b> (@${v.uploader}) | Duration: ${durationFormatted}
                </div>
              </div>
              <div style="text-align: right; display: flex; flex-direction: column; gap: 4px;">
                <div class="mono" style="font-size:11px; color:var(--muted);"><span style="color:#64748b;">Summary:</span> ${pipe.models.summary.split(':').pop()}</div>
                <div class="mono" style="font-size:11px; color:var(--muted);"><span style="color:#64748b;">Subtitles:</span> ${pipe.models.subtitles.split(':').pop()}</div>
                <div class="mono" style="font-size:11px; color:var(--muted);"><span style="color:#64748b;">Lyrics:</span> ${pipe.models.lyrics.split(':').pop()}</div>
              </div>
            </div>

            <!-- Consistent 4-Stage Breakdown with Stacked Bars: In Progress, Done, Total -->
            <div class="grid">
              <!-- Stage 1: Summary -->
              <div class="stage-card">
                <div>
                  <b style="font-size: 13px;">1. Whole-Video Summary</b>
                  <div style="display:flex; justify-content:space-between; font-size: 11px; margin-top: 6px;">
                    <span>
                      ${sum.in_progress > 0 ? `<span style="color:var(--blue); font-weight:600;">${sum.in_progress} In Progress</span> · ` : ''}
                      <b style="color:var(--green);">${sum.done} Done</b>
                    </span>
                    <span style="color:var(--muted);">Total: ${sum.total} (${sumDonePct}%)</span>
                  </div>
                  <div class="progress-bar">
                    <div class="progress-fill-done" style="width: ${sumDonePct}%;"></div>
                    <div class="progress-fill-prog" style="width: ${sumProgPct}%;"></div>
                  </div>
                </div>
                <div style="font-size: 11px; color: var(--muted); margin-top: 4px;">
                  ${sum.done > 0 ? 'Macro knowledge index & chapters ready.' : (sum.in_progress > 0 ? `Running by worker <b>${sum.worker_id}</b> (${sum.elapsed_s}s)` : 'Waiting for worker pickup.')}
                </div>
              </div>

              <!-- Stage 2: Re-encoding -->
              <div class="stage-card">
                <div>
                  <b style="font-size: 13px;">2. Chunk Compression</b>
                  <div style="display:flex; justify-content:space-between; font-size: 11px; margin-top: 6px;">
                    <span>
                      ${reenc.in_progress > 0 ? `<span style="color:var(--blue); font-weight:600;">${reenc.in_progress} In Progress</span> · ` : ''}
                      <b style="color:var(--green);">${reenc.done} Done</b>
                    </span>
                    <span style="color:var(--muted);">Total: ${reenc.total} (${reencDonePct}%)</span>
                  </div>
                  <div class="progress-bar">
                    <div class="progress-fill-done" style="width: ${reencDonePct}%;"></div>
                    <div class="progress-fill-prog" style="width: ${reencProgPct}%;"></div>
                  </div>
                </div>
                <div style="font-size: 11px; color: var(--muted); margin-top: 4px;">
                  ${reenc.pending > 0 ? `${reenc.pending} chunk(s) pending compression` : 'All chunks compressed and ready.'}
                </div>
              </div>

              <!-- Stage 3: Subtitles -->
              <div class="stage-card">
                <div>
                  <b style="font-size: 13px;">3. Subtitles Transcription</b>
                  <div style="display:flex; justify-content:space-between; font-size: 11px; margin-top: 6px;">
                    <span>
                      ${sub.in_progress > 0 ? `<span style="color:var(--blue); font-weight:600;">${sub.in_progress} In Progress</span> · ` : ''}
                      <b style="color:var(--green);">${sub.done} Done</b>
                    </span>
                    <span style="color:var(--muted);">Total: ${sub.total} (${subDonePct}%)</span>
                  </div>
                  <div class="progress-bar">
                    <div class="progress-fill-done" style="width: ${subDonePct}%;"></div>
                    <div class="progress-fill-prog" style="width: ${subProgPct}%;"></div>
                  </div>
                </div>
                <div style="font-size: 11px; color: var(--muted); margin-top: 4px;">
                  ${sub.in_progress > 0
                    ? `${sub.in_progress} chunk(s) actively running on workers`
                    : (sub.done === sub.total && sub.total > 0
                        ? 'All subtitle chunks transcribed.'
                        : (sum.done === 0 ? 'Waiting for whole-video summary to complete.' : 'Waiting for worker pickup.'))}
                </div>
              </div>

              <!-- Stage 4: Lyrics -->
              <div class="stage-card">
                <div>
                  <b style="font-size: 13px;">4. Lyrics Refinement</b>
                  <div style="display:flex; justify-content:space-between; font-size: 11px; margin-top: 6px;">
                    <span>
                      ${lyr.status === 'pending'
                        ? '<span style="color:var(--muted);">Waiting for Subtitles</span>'
                        : (lyr.status === 'skipped'
                            ? '<span style="color:var(--muted);">No Singing Detected (Skipped)</span>'
                            : `${lyr.in_progress > 0 ? `<span style="color:var(--blue); font-weight:600;">${lyr.in_progress} In Progress</span> · ` : ''}<b style="color:var(--green);">${lyr.done} Done</b>`)}
                    </span>
                    <span style="color:var(--muted);">${lyr.total > 0 ? `Total: ${lyr.total} (${lyrDonePct}%)` : 'Total: 0'}</span>
                  </div>
                  <div class="progress-bar">
                    <div class="progress-fill-done" style="width: ${lyrDonePct}%;"></div>
                    <div class="progress-fill-prog" style="width: ${lyrProgPct}%;"></div>
                  </div>
                </div>
                <div style="font-size: 11px; color: var(--muted); margin-top: 4px;">
                  ${
                    lyr.status === 'pending'
                      ? 'Vocal performances will be detected during transcription.'
                      : (lyr.status === 'skipped'
                          ? 'No vocal singing detected across all chunks.'
                          : (lyr.in_progress > 0
                              ? lyr.active_leases.map(l => `Refining with <b>${l.worker_id}</b> (${l.elapsed_s}s)`).join(', ')
                              : (lyr.done === lyr.total ? 'All singing chunks refined.' : 'Waiting for worker pickup.')))
                  }
                </div>
              </div>
            </div>

            <div style="font-size: 12px; margin-top: 8px;">
              <b>Active Leases:</b>
              ${pipe.tasks.filter(t => t.status === 'claimed').map(t => {
                const parts = t.task_id.split('::');
                const lastPart = parts.pop();
                const taskLabel = lastPart === t.type ? t.type : `${t.type} ${lastPart}`;
                return `<span class="badge badge-claim mono">${taskLabel} (${t.worker_id}) [${t.elapsed_s}s]</span>`;
              }).join(' ') || '<span style="color:var(--muted);">No active worker leases at this moment</span>'}
            </div>
          </div>
        `;
      } else {
        let statusTitle = `${pipe.name} is currently idle`;
        let statusDesc = 'Waiting for next video from intake sources...';

        if (pipe.status === 'preparing') {
          statusTitle = `${pipe.name} is preparing media`;
          statusDesc = 'Downloading video stream or generating segment chunks...';
        } else if (pipe.status === 'finalizing') {
          statusTitle = `${pipe.name} is finalizing subtitles`;
          statusDesc = 'Stitching completed subtitle segments and committing to Git...';
        } else if (pipe.status === 'push_failed') {
          statusTitle = `${pipe.name} encountered a Git push failure`;
          statusDesc = 'Subtitles completed on disk. Retrying Git synchronization automatically on next cycle...';
        }

        activeSection = `
          <div class="card" style="margin-bottom: 20px; padding: 36px; text-align: center;">
            <div style="margin-bottom: 10px;">
              <span class="badge badge-${pipe.status || 'idle'}">${pipe.status || 'idle'}</span>
            </div>
            <h3 style="margin-bottom: 6px;">${statusTitle}</h3>
            <p style="color: var(--muted); margin: 0;">${statusDesc}</p>
            <div class="mono" style="font-size: 12px; margin-top: 14px; color: var(--muted);">
              Models: ${pipe.models.summary.split(':').pop()} / ${pipe.models.subtitles.split(':').pop()} / ${pipe.models.lyrics.split(':').pop()}
            </div>
          </div>
        `;
      }

      const queueTable = (pipe.next_in_queue || []).map(q => `
        <tr>
          <td>${q.title}</td>
          <td><b>${q.channel_name || q.uploader}</b> <span class="mono" style="color:var(--muted);">(@${q.uploader})</span></td>
          <td><span class="badge badge-${q.source_type}">${q.source_type}</span></td>
        </tr>
      `).join('') || '<tr><td colspan="3" style="color: var(--muted);">No pending videos in queue for this pipeline.</td></tr>';

      document.getElementById('tab-content').innerHTML = `
        ${activeSection}
        <div class="card">
          <h3>Next In Queue for ${pipe.name}</h3>
          <table>
            <thead><tr><th>Video Title</th><th>Channel / Uploader</th><th>Source</th></tr></thead>
            <tbody>${queueTable}</tbody>
          </table>
        </div>
      `;
    }

    async function updateDashboard() {
      try {
        const res = await fetch('/api/v1/dashboard/data');
        if (!res.ok) return;
        currentData = await res.json();
        document.getElementById('server-time').innerText = 'Server Time: ' + currentData.server_time;

        renderPipelineView();

        const aBody = document.querySelector('#activity-table tbody');
        aBody.innerHTML = (currentData.recent_activity || []).slice(0, 30).map((a, idx) => {
          const parts = a.task_id.split('::');
          const cleanTaskId = parts.length > 2 ? parts.slice(2).join(' ') : (parts[1] || a.task_id);
          return `
            <tr>
              <td class="mono">${a.time_str}</td>
              <td class="mono"><b>${a.worker_id}</b></td>
              <td class="mono">${a.pipeline}</td>
              <td class="mono">${cleanTaskId}</td>
              <td><span class="badge badge-${a.event_type}">${a.event_type} ${a.duration_s ? a.duration_s + 's' : ''}</span></td>
              <td>${a.detail ? `<button class="expand-btn" onclick="toggleDetail(${idx})">Details</button>` : '-'}</td>
            </tr>
            ${a.detail ? `
              <tr class="detail-row" id="detail-${idx}">
                <td colspan="6"><div class="detail-box ${a.event_type === 'fail' ? 'detail-box-fail' : ''}">${a.detail}</div></td>
              </tr>
            ` : ''}
          `;
        }).join('') || '<tr><td colspan="6" style="color: var(--muted);">No worker activity recorded yet.</td></tr>';

      } catch (err) {
        console.error(err);
      }
    }

    setInterval(updateDashboard, 60000);
    updateDashboard();
  </script>
</body>
</html>
"""  # noqa: E501

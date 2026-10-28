"""Embedded live WebUI dashboard HTML template for ai-sub-server."""

from __future__ import annotations

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Sub Coordinator</title>
  <style>
    :root {
      --bg: #0f172a; --panel: #1e293b; --text: #f8fafc; --muted: #94a3b8;
      --accent: #38bdf8; --border: #334155; --green: #22c55e; --yellow: #eab308;
      --red: #ef4444; --blue: #3b82f6; --purple: #a855f7; --pink: #ec4899;
    }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
    h1, h2, h3, h4 { margin-top: 0; font-weight: 600; }
    .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 20px; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }
    .stage-card { background: #0f172a; border: 1px solid var(--border); border-radius: 6px; padding: 14px; }
    .progress-bar { background: #1e293b; border-radius: 9999px; height: 8px; overflow: hidden; margin: 8px 0; border: 1px solid var(--border); }
    .progress-fill { background: var(--accent); height: 100%; width: 0%; transition: width 0.3s; }
    .progress-fill-enc { background: var(--purple); height: 100%; width: 0%; transition: width 0.3s; }
    .progress-fill-lyr { background: var(--pink); height: 100%; width: 0%; transition: width 0.3s; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
    .badge-claim { background: #1e3a8a; color: #93c5fd; }
    .badge-complete { background: #14532d; color: #86efac; }
    .badge-fail { background: #7f1d1d; color: #fca5a5; }
    .badge-timeout { background: #713f12; color: #fde047; }
    .badge-pending { background: #334155; color: #cbd5e1; }
    .btn { background: #334155; color: var(--text); border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500; }
    .btn:hover { background: #475569; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-weight: 500; }
    .mono { font-family: monospace; font-size: 12px; }
    
    /* Tabs */
    .tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
    .tab-btn { background: transparent; border: none; color: var(--muted); padding: 10px 18px; font-size: 14px; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; }
    .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

    /* Expandable activity details */
    .expand-btn { background: none; border: 1px solid var(--border); color: var(--muted); padding: 2px 6px; border-radius: 4px; font-size: 11px; cursor: pointer; }
    .expand-btn:hover { color: var(--text); }
    .detail-row { background: #0f172a; font-family: monospace; font-size: 12px; color: #cbd5e1; display: none; }
    .detail-box { padding: 12px 16px; white-space: pre-wrap; word-break: break-all; margin: 0; line-height: 1.5; border-radius: 4px; }
    .detail-box-fail { background: rgba(239, 68, 68, 0.1); border-left: 3px solid var(--red); color: #fca5a5; }
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
      <button class="btn" onclick="rescanSources()">🔍 Rescan Sources</button>
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

    async function rescanSources() {
      try {
        await fetch('/api/v1/sources/rescan', { method: 'POST' });
        alert('Rescan triggered in background. Updating UI shortly...');
        setTimeout(updateDashboard, 1500);
      } catch (err) {
        alert('Error triggering rescan: ' + err);
      }
    }

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
      tabs.innerHTML = currentData.pipelines.map((p, idx) => `
        <button class="tab-btn ${idx === selectedPipelineIndex ? 'active' : ''}" onclick="selectTab(${idx})">
          ${p.name} ${p.idle ? '<span style="opacity:0.6;">(Idle)</span>' : '<span style="color:var(--green);">●</span>'}
        </button>
      `).join('');

      const v = pipe.active_video;
      const st = pipe.stages;

      let activeSection = '';
      if (v && st) {
        const reenc = st.reencoding;
        const sub = st.subtitles;
        const lyr = st.lyrics;
        const sum = st.summary;

        const reencPct = reenc.total_chunks > 0 ? Math.round((reenc.reencoded / reenc.total_chunks) * 100) : 100;
        const sumStatusBadge = sum.status === 'completed' ? 'badge-complete' : (sum.status === 'claimed' ? 'badge-claim' : 'badge-pending');

        activeSection = `
          <div class="card" style="margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 16px;">
              <div>
                <span class="badge badge-complete" style="margin-bottom:6px;">Active Video</span>
                <h3 style="margin-bottom: 4px;">${v.title}</h3>
                <div style="color: var(--muted); font-size: 13px;">
                  Channel: <b>${v.channel_name || v.uploader}</b> (@${v.uploader}) | Duration: ${Math.round(v.duration_ms / 60000)} minutes
                </div>
              </div>
              <div style="text-align: right;">
                <div class="mono" style="font-size:12px; color:var(--muted);">Summary: ${pipe.models.summary.split(':').pop()}</div>
                <div class="mono" style="font-size:12px; color:var(--muted);">Subtitles: ${pipe.models.subtitles.split(':').pop()}</div>
                <div class="mono" style="font-size:12px; color:var(--muted);">Lyrics: ${pipe.models.lyrics.split(':').pop()}</div>
              </div>
            </div>

            <!-- Granular Stage Breakdown -->
            <div class="grid">
              <!-- Stage 1: Summary -->
              <div class="stage-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                  <b>1. Whole-Video Summary</b>
                  <span class="badge ${sumStatusBadge}">
                    ${sum.status === 'completed' ? 'Completed' : (sum.status === 'claimed' ? 'In Progress' : 'Pending')}
                  </span>
                </div>
                <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">
                  ${sum.status === 'completed' ? 'Macro knowledge index & chapters ready.' : (sum.status === 'claimed' ? `Running by worker <b>${sum.worker_id}</b> (${sum.elapsed_s}s)` : 'Waiting for worker pickup.')}
                </div>
              </div>

              <!-- Stage 2: Re-encoding -->
              <div class="stage-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                  <b>2. Chunk Compression</b>
                  <span class="badge ${reenc.pending === 0 ? 'badge-complete' : 'badge-claim'}">
                    ${reenc.pending === 0 ? 'Ready' : 'Compressing'}
                  </span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size: 11px; color: var(--muted); margin-top: 4px;">
                  <span>${reenc.reencoded}/${reenc.total_chunks} chunks ready</span>
                  <span>${reencPct}%</span>
                </div>
                <div class="progress-bar"><div class="progress-fill-enc" style="width: ${reencPct}%;"></div></div>
                <div style="font-size: 11px; color: var(--muted);">
                  ${reenc.pending > 0 ? `${reenc.pending} pending FFmpeg compression` : 'All chunks ready for workers.'}
                </div>
              </div>

              <!-- Stage 3: Subtitles -->
              <div class="stage-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                  <b>3. Subtitles Transcription</b>
                  <span class="badge ${
                    sub.total === 0 ? 'badge-pending' : (
                      sub.completed === sub.total ? 'badge-complete' : (
                        sub.claimed > 0 ? 'badge-claim' : 'badge-pending'
                      )
                    )
                  }">
                    ${
                      sub.total === 0 ? 'Waiting' : (
                        sub.completed === sub.total ? 'Completed' : (
                          sub.claimed > 0 ? 'In Progress' : 'Pending'
                        )
                      )
                    }
                  </span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size: 11px; color: var(--muted); margin-top: 4px;">
                  <span>${sub.completed}/${sub.total} chunks transcribed</span>
                  <span>${sub.percentage}%</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" style="width: ${sub.percentage}%;"></div></div>
                <div style="font-size: 11px; color: var(--muted);">
                  ${sub.claimed > 0 ? `${sub.claimed} chunks currently running with workers` : (sub.completed === sub.total && sub.total > 0 ? 'All chunks transcribed.' : 'Waiting for worker pickup.')}
                </div>
              </div>

              <!-- Stage 4: Lyrics -->
              <div class="stage-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                  <b>4. Lyrics Refinement</b>
                  <span class="badge ${
                    !lyr.has_singing ? 'badge-pending' : (
                      lyr.status === 'completed' ? 'badge-complete' : (
                        lyr.status === 'in_progress' ? 'badge-claim' : 'badge-pending'
                      )
                    )
                  }">
                    ${
                      !lyr.has_singing ? 'Skipped' : (
                        lyr.status === 'completed' ? 'Completed' : (
                          lyr.status === 'in_progress' ? 'In Progress' : 'Pending'
                        )
                      )
                    }
                  </span>
                </div>
                ${lyr.has_singing ? `
                  <div style="display:flex; justify-content:space-between; font-size: 11px; color: var(--muted); margin-top: 4px;">
                    <span>${lyr.completed}/${lyr.total} singing chunks refined</span>
                    <span>${Math.round((lyr.completed / lyr.total) * 100)}%</span>
                  </div>
                  <div class="progress-bar"><div class="progress-fill-lyr" style="width: ${Math.round((lyr.completed / lyr.total) * 100)}%;"></div></div>
                  <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">
                    ${lyr.claimed > 0 
                      ? lyr.active_leases.map(l => `Refining with <b>${l.worker_id}</b> (${l.elapsed_s}s)`).join(', ')
                      : (lyr.completed === lyr.total ? 'All singing chunks refined.' : 'Waiting for worker pickup.')
                    }
                  </div>
                ` : `
                  <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">
                    No vocal singing detected in transcript.
                  </div>
                `}
              </div>
            </div>

            <div style="font-size: 12px; margin-top: 8px;">
              <b>Active Leases:</b>
              ${pipe.tasks.filter(t => t.status === 'claimed').map(t => `
                <span class="badge badge-claim mono">${t.task_id.split('::').pop()} (${t.worker_id}) [${t.elapsed_s}s]</span>
              `).join(' ') || '<span style="color:var(--muted);">No active leases at this moment</span>'}
            </div>
          </div>
        `;
      } else {
        activeSection = `
          <div class="card" style="margin-bottom: 20px; padding: 30px; text-align: center; color: var(--muted);">
            <h3>${pipe.name} is currently idle</h3>
            <p>Waiting for next video from intake sources...</p>
            <div class="mono" style="font-size: 12px; margin-top: 8px;">
              Models: ${pipe.models.summary.split(':').pop()} / ${pipe.models.subtitles.split(':').pop()} / ${pipe.models.lyrics.split(':').pop()}
            </div>
          </div>
        `;
      }

      const queueTable = (pipe.next_in_queue || []).map(q => `
        <tr>
          <td>${q.title}</td>
          <td><b>${q.channel_name || q.uploader}</b> <span class="mono" style="color:var(--muted);">(@${q.uploader})</span></td>
          <td><span class="badge badge-${q.source_type === 'youtube' ? 'claim' : 'complete'}">${q.source_type}</span></td>
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
        aBody.innerHTML = (currentData.recent_activity || []).slice(0, 30).map((a, idx) => `
          <tr>
            <td class="mono">${a.time_str}</td>
            <td class="mono"><b>${a.worker_id}</b></td>
            <td class="mono">${a.pipeline}</td>
            <td class="mono">${a.task_id.split('::').slice(1).join('::')}</td>
            <td><span class="badge badge-${a.event_type}">${a.event_type} ${a.duration_s ? a.duration_s + 's' : ''}</span></td>
            <td>${a.detail ? `<button class="expand-btn" onclick="toggleDetail(${idx})">Details</button>` : '-'}</td>
          </tr>
          ${a.detail ? `
            <tr class="detail-row" id="detail-${idx}">
              <td colspan="6"><div class="detail-box ${a.event_type === 'fail' ? 'detail-box-fail' : ''}">${a.detail}</div></td>
            </tr>
          ` : ''}
        `).join('') || '<tr><td colspan="6" style="color: var(--muted);">No worker activity recorded yet.</td></tr>';

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

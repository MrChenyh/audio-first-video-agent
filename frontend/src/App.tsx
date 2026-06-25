import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock3,
  FileAudio,
  Gauge,
  Image as ImageIcon,
  Link as LinkIcon,
  Loader2,
  MessageSquareText,
  Play,
  SearchCheck,
  Send,
  UploadCloud
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type JobStatus = "queued" | "running" | "succeeded" | "failed";
type SourceMode = "file" | "url";

type Job = {
  job_id: string;
  question: string;
  status: JobStatus;
  progress: number;
  current_node: string;
  error?: string | null;
};

type Probe = {
  type?: string;
  event_index?: number;
  label?: string;
  window_start?: number;
  window_end?: number;
  question?: string;
  expected_visuals?: string[];
  audio_evidence?: string;
  user_question?: string;
};

type Frame = {
  time: number;
  filename: string;
  url: string;
  reason: string;
  probe?: Probe | null;
  observation?: {
    scene: string;
    objects: string[];
    actions: string[];
    visible_text: string[];
    audio_alignment: "match" | "conflict" | "uncertain";
    visual_target?: string;
    evidence_assessment?: string;
    notes: string;
  } | null;
};

type TranscriptSegment = {
  start: number;
  end: number;
  speaker?: string;
  text: string;
  confidence?: number | null;
  source?: string;
};

type Result = {
  job_id?: string;
  question: string;
  partial?: boolean;
  status?: JobStatus;
  progress?: number;
  current_node?: string;
  answer?: {
    direct_answer: string;
    summary: string;
    evidence_refs: string[];
    uncertainties: string[];
  } | null;
  timeline: Array<{
    time: number;
    end_time?: number | null;
    label: string;
    evidence: string;
    expected_visuals: string[];
    visual_question?: string;
  }>;
  transcript_segments?: TranscriptSegment[];
  frames: Frame[];
  prediction_checks: Array<{
    window_start: number;
    window_end: number;
    hypothesis: string;
    status: "match" | "conflict" | "uncertain";
    conflict_score: number;
    evidence: string;
    expected_evidence?: string[];
    source_event?: string;
  }>;
  metadata: {
    duration_seconds?: number;
    width?: number;
    height?: number;
    mock_mode?: boolean;
    fast_mode?: boolean;
    vision_provider?: string;
    vision_model?: string;
    vision_request_count?: number;
  };
  transcription_status?: {
    status?: string;
    reason?: string;
    method?: string;
    api_attempted?: boolean;
    chat_audio_attempted?: boolean;
    local_attempted?: boolean;
    api_error?: string;
    local_error?: string;
    segment_count?: number;
    attempts?: Array<Record<string, unknown>>;
  };
};

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  evidence?: string[];
  coverage?: string;
};

type CoverageRow = {
  start: string;
  end?: string;
  title: string;
  text: string;
};

const nodeLabels: Record<string, string> = {
  queued: "排队中",
  starting: "启动任务",
  download_url: "下载视频",
  ingest_video: "读取视频",
  extract_audio: "抽取音频",
  transcribe_audio: "识别音频",
  build_audio_world_model: "构建音频先验",
  generate_frame_candidates: "筛选候选画面",
  plan_keyframes: "规划关键画面",
  extract_keyframes: "抽取关键画面",
  observe_frames: "观察关键画面",
  predict_next_events: "推演后续片段",
  verify_predictions: "检查覆盖情况",
  synthesize_answer: "生成总结",
  complete: "完成",
  failed: "失败"
};

export function App() {
  const [sourceMode, setSourceMode] = useState<SourceMode>("file");
  const [video, setVideo] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [question, setQuestion] = useState("这个视频主要发生了什么？请按时间线总结，并给出关键证据。");
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [followup, setFollowup] = useState("");
  const [asking, setAsking] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const partialTimerRef = useRef<number | null>(null);

  const statusLabel = useMemo(() => {
    if (!job) return sourceMode === "file" ? "等待上传" : "等待 URL";
    if (job.status === "failed") return "分析失败";
    if (job.status === "succeeded") return "分析完成";
    return labelForNode(job.current_node);
  }, [job, sourceMode]);

  const sourceVideoUrl = job ? `/api/jobs/${job.job_id}/source` : null;

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      stopPartialPolling();
    };
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (sourceMode === "file" && !video) {
      setError("请选择一个视频文件。");
      return;
    }
    if (sourceMode === "url" && !videoUrl.trim()) {
      setError("请输入一个视频 URL。");
      return;
    }
    setSubmitting(true);
    setError(null);
    setResult(null);
    setJob(null);
    setChat([]);

    try {
      const payload = sourceMode === "file" ? await createFileJob() : await createUrlJob();
      subscribe(payload.job_id);
    } catch (err) {
      setError(errorMessage(err, sourceMode === "url" ? "URL 创建任务失败。" : "上传失败。"));
    } finally {
      setSubmitting(false);
    }
  }

  async function createFileJob() {
    const body = new FormData();
    if (video) body.append("video", video);
    body.append("question", question);
    const response = await fetch("/api/jobs", { method: "POST", body });
    if (!response.ok) throw new Error(await response.text());
    return (await response.json()) as { job_id: string };
  }

  async function createUrlJob() {
    const response = await fetch("/api/jobs/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: videoUrl.trim(), question })
    });
    if (!response.ok) throw new Error(await response.text());
    return (await response.json()) as { job_id: string };
  }

  function subscribe(jobId: string) {
    eventSourceRef.current?.close();
    stopPartialPolling();
    const source = new EventSource(`/api/jobs/${jobId}/events`);
    eventSourceRef.current = source;
    startPartialPolling(jobId);
    source.onmessage = async (event) => {
      const nextJob = JSON.parse(event.data) as Job;
      setJob(nextJob);
      if (nextJob.status === "succeeded") {
        source.close();
        stopPartialPolling();
        const response = await fetch(`/api/jobs/${nextJob.job_id}/result`);
        if (response.ok) setResult((await response.json()) as Result);
      }
      if (nextJob.status === "failed") {
        source.close();
        stopPartialPolling();
        setError(nextJob.error ?? "任务失败。");
      }
    };
    source.onerror = () => {
      source.close();
      stopPartialPolling();
      setError("进度连接中断，请刷新任务状态。");
    };
  }

  function startPartialPolling(jobId: string) {
    const poll = async () => {
      try {
        const response = await fetch(`/api/jobs/${jobId}/partial`);
        if (response.ok) {
          const payload = (await response.json()) as Result;
          setResult((current) => mergePartialResult(current, payload));
        }
      } catch {
        // SSE still owns fatal progress errors; partial polling is best-effort.
      }
    };
    void poll();
    partialTimerRef.current = window.setInterval(poll, 1600);
  }

  function stopPartialPolling() {
    if (partialTimerRef.current !== null) {
      window.clearInterval(partialTimerRef.current);
      partialTimerRef.current = null;
    }
  }

  async function askFollowup(event: FormEvent) {
    event.preventDefault();
    const text = followup.trim();
    const jobId = job?.job_id ?? result?.job_id;
    if (!text || !jobId) return;
    setChat((items) => [...items, { role: "user", text }]);
    setFollowup("");
    setAsking(true);
    try {
      const response = await fetch(`/api/jobs/${jobId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text })
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as {
        answer: { answer: string; evidence_refs?: string[]; coverage_note?: string };
      };
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: payload.answer.answer,
          evidence: payload.answer.evidence_refs ?? [],
          coverage: payload.answer.coverage_note
        }
      ]);
    } catch (err) {
      setChat((items) => [...items, { role: "assistant", text: errorMessage(err, "追问失败。") }]);
    } finally {
      setAsking(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="topbar">
          <div>
            <p className="eyebrow">LangGraph multimodal harness</p>
            <h1>Audio-First Video Agent</h1>
          </div>
          <div className="status-pill" title="当前任务状态">
            {job?.status === "running" || job?.status === "queued" ? (
              <Loader2 className="spin" size={16} />
            ) : (
              <Activity size={16} />
            )}
            <span>{statusLabel}</span>
          </div>
        </div>

        <div className="main-grid">
          <form className="upload-panel" onSubmit={submit}>
            <div className="source-tabs" role="tablist" aria-label="视频来源">
              <button type="button" className={sourceMode === "file" ? "active" : ""} onClick={() => setSourceMode("file")}>
                <UploadCloud size={16} /> 文件
              </button>
              <button type="button" className={sourceMode === "url" ? "active" : ""} onClick={() => setSourceMode("url")}>
                <LinkIcon size={16} /> URL
              </button>
            </div>

            {sourceMode === "file" ? (
              <label className="drop-zone">
                <input
                  type="file"
                  accept="video/*"
                  onChange={(event) => setVideo(event.target.files?.[0] ?? null)}
                />
                <UploadCloud size={28} />
                <span>{video ? video.name : "选择 1 到 30 分钟的视频"}</span>
              </label>
            ) : (
              <label className="url-box">
                <span>
                  <LinkIcon size={16} /> 视频 URL
                </span>
                <input
                  type="url"
                  value={videoUrl}
                  onChange={(event) => setVideoUrl(event.target.value)}
                  placeholder="https://example.com/video.mp4"
                />
                <small>直链 mp4/webm/mov 可直接下载；站点页面链接会尝试 yt-dlp。</small>
              </label>
            )}

            <label className="question-box">
              <span>
                <MessageSquareText size={16} /> 问题
              </span>
              <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={5} />
            </label>

            <button className="primary-action" type="submit" disabled={submitting || (sourceMode === "file" ? !video : !videoUrl.trim())}>
              {submitting ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              <span>{submitting ? "提交中" : "开始分析"}</span>
            </button>
          </form>

          <section className="progress-panel">
            <div className="progress-header">
              <Gauge size={18} />
              <span>{job ? `${job.progress}%` : "0%"}</span>
            </div>
            <div className="progress-track">
              <div style={{ width: `${job?.progress ?? 0}%` }} />
            </div>
            <div className="stage-list">
              {[
                "download_url",
                "transcribe_audio",
                "generate_frame_candidates",
                "plan_keyframes",
                "observe_frames",
                "synthesize_answer"
              ].map((node) => (
                <div className={isActiveNode(job?.current_node, node) ? "stage active" : "stage"} key={node}>
                  <Clock3 size={14} />
                  <span>{nodeLabels[node]}</span>
                </div>
              ))}
            </div>
            {sourceVideoUrl && (
              <video className="source-preview" src={sourceVideoUrl} controls preload="metadata" />
            )}
            {error && (
              <div className="error-line" role="alert">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}
            {result?.metadata.mock_mode && (
              <div className="mock-line">
                <AlertCircle size={16} />
                <span>当前是 mock 模式，适合验证界面和流程。</span>
              </div>
            )}
          </section>
        </div>

        {result && <ResultView result={result} chat={chat} followup={followup} asking={asking} setFollowup={setFollowup} askFollowup={askFollowup} />}
      </section>
    </main>
  );
}

function ResultView({
  result,
  chat,
  followup,
  asking,
  setFollowup,
  askFollowup
}: {
  result: Result;
  chat: ChatMessage[];
  followup: string;
  asking: boolean;
  setFollowup: (value: string) => void;
  askFollowup: (event: FormEvent) => void;
}) {
  const observedFrames = result.frames.filter((frame) => frame.observation);
  const pendingFrames = result.frames.filter((frame) => !frame.observation);
  const answerReady = Boolean(result.answer?.direct_answer);

  return (
    <section className="result-grid">
      <div className="answer-band">
        <div className="section-title">
          <CheckCircle2 size={18} />
          <h2>{answerReady ? "最终回答" : "实时观察"}</h2>
        </div>
        {answerReady ? (
          <>
            <p className="direct-answer">{result.answer?.direct_answer}</p>
            <p>{result.answer?.summary}</p>
          </>
        ) : (
          <p className="empty">音频和关键画面正在进入分析链路，已完成的观察会先显示在下面。</p>
        )}
        <div className="metric-row">
          <span>{result.metadata.duration_seconds ? formatTime(result.metadata.duration_seconds) : "--:--"} 视频时长</span>
          <span>{result.transcript_segments?.length ?? 0} 段音频</span>
          <span>{observedFrames.length}/{result.frames.length} 帧已观察</span>
          {result.metadata.vision_provider && <span>{result.metadata.vision_provider}</span>}
        </div>
      </div>

      <div className="timeline-band">
        <div className="section-title">
          <Activity size={18} />
          <h2>音频时间线</h2>
        </div>
        {result.transcription_status && (
          <div className="audio-status">
            <strong>{transcriptionLabel(result.transcription_status.status)} / {result.transcription_status.method ?? "unknown"}</strong>
            <span>{result.transcription_status.reason ?? "没有返回转写状态。"}</span>
            {result.transcription_status.local_error && <small>{result.transcription_status.local_error}</small>}
          </div>
        )}
        {Boolean(result.transcript_segments?.length) && (
          <div className="transcript-list compact-scroll">
            <div className="section-title compact">
              <FileAudio size={16} />
              <h3>音频转写</h3>
            </div>
            {result.transcript_segments?.slice(0, 18).map((segment, index) => (
              <article className="transcript-row" key={`${segment.start}-${index}`}>
                <span className="time">{formatTime(segment.start)}</span>
                <p>{segment.text}</p>
              </article>
            ))}
          </div>
        )}
        <div className="timeline">
          {result.timeline.length === 0 && <p className="empty">没有可用音频时间线，已退化为稀疏视觉采样。</p>}
          {result.timeline.map((event, index) => (
            <article className="timeline-item" key={`${event.time}-${index}`}>
              <span className="time">{formatTime(event.time)}</span>
              <div>
                <h3>{event.label}</h3>
                <p>{event.evidence}</p>
                {event.expected_visuals.length > 0 && <small>{event.expected_visuals.join(" / ")}</small>}
              </div>
            </article>
          ))}
        </div>
      </div>

      <div className="frames-band">
        <div className="section-title">
          <SearchCheck size={18} />
          <h2>关键画面证据</h2>
        </div>
        <div className="frame-grid">
          {result.frames.map((frame) => (
            <article className="frame-card" key={frame.filename}>
              <img src={frame.url} alt={`frame at ${formatTime(frame.time)}`} loading="lazy" />
              <div>
                <span className={frame.observation ? "evidence-pill observed" : "evidence-pill pending"}>
                  {frame.observation ? "已观察" : "排队中"}
                </span>
                <h3>{formatTime(frame.time)}</h3>
                <p className="target-text">{frame.observation?.visual_target ?? frame.probe?.question ?? frame.reason}</p>
                <p>{frame.observation?.scene ?? frame.reason}</p>
                {frame.observation?.evidence_assessment && <small>{frame.observation.evidence_assessment}</small>}
              </div>
            </article>
          ))}
        </div>
        {pendingFrames.length > 0 && <p className="coverage-note">还有 {pendingFrames.length} 个关键画面正在等待观察。</p>}
      </div>

      <div className="coverage-band">
        <div className="section-title">
          <ImageIcon size={18} />
          <h2>覆盖情况</h2>
        </div>
        <div className="coverage-list">
          {coverageRows(result).map((row, index) => (
            <article className="coverage-row" key={`${row.start}-${index}`}>
              <span className="time">{row.start}{row.end ? `-${row.end}` : ""}</span>
              <div>
                <h3>{row.title}</h3>
                <p>{row.text}</p>
              </div>
            </article>
          ))}
        </div>
      </div>

      <div className="chat-band">
        <div className="section-title">
          <MessageSquareText size={18} />
          <h2>继续追问</h2>
        </div>
        <div className="chat-list">
          {chat.length === 0 && <p className="empty">分析完成前也可以先输入问题；回答会基于当前已有的音频和关键画面。</p>}
          {chat.map((message, index) => (
            <article className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
              <p>{message.text}</p>
              {Boolean(message.evidence?.length) && <small>{message.evidence?.join(" / ")}</small>}
              {message.coverage && <small>{message.coverage}</small>}
            </article>
          ))}
        </div>
        <form className="followup-form" onSubmit={askFollowup}>
          <input value={followup} onChange={(event) => setFollowup(event.target.value)} placeholder="继续问这个视频里的细节" />
          <button type="submit" disabled={asking || !followup.trim()} aria-label="发送追问">
            {asking ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          </button>
        </form>
      </div>
    </section>
  );
}

function labelForNode(node: string) {
  if (node.startsWith("observe_frames")) return `观察关键画面 ${node.replace("observe_frames", "").trim()}`.trim();
  return nodeLabels[node] ?? node;
}

function isActiveNode(current: string | undefined, node: string) {
  if (!current) return false;
  if (node === "observe_frames") return current.startsWith("observe_frames");
  return current === node;
}

function transcriptionLabel(status?: string) {
  if (status === "ok") return "已识别";
  if (status === "mock") return "模拟识别";
  if (status === "empty") return "未识别到文本";
  if (status === "skipped") return "未处理";
  return status ?? "unknown";
}

function coverageRows(result: Result): CoverageRow[] {
  const rows = result.prediction_checks.slice(0, 8).map((check) => ({
    start: formatTime(check.window_start),
    end: formatTime(check.window_end),
    title: check.source_event || "片段覆盖",
    text: check.evidence || check.hypothesis
  }));
  if (rows.length > 0) return rows;
  return result.frames.slice(0, 8).map((frame) => ({
    start: formatTime(frame.time),
    title: frame.observation ? "已观察关键画面" : "计划观察关键画面",
    text: frame.observation?.evidence_assessment || frame.observation?.scene || frame.reason
  }));
}

function mergePartialResult(current: Result | null, next: Result) {
  if (!current) return next;
  if (!next.partial) return next;
  return {
    ...current,
    ...next,
    answer: current.answer ?? next.answer,
    timeline: next.timeline.length ? next.timeline : current.timeline,
    transcript_segments: next.transcript_segments?.length ? next.transcript_segments : current.transcript_segments,
    frames: next.frames.length ? next.frames : current.frames,
    prediction_checks: next.prediction_checks.length ? next.prediction_checks : current.prediction_checks,
    metadata: { ...current.metadata, ...next.metadata },
    transcription_status: next.transcription_status ?? current.transcription_status
  };
}

function errorMessage(err: unknown, fallback: string) {
  if (!(err instanceof Error)) return fallback;
  try {
    const parsed = JSON.parse(err.message) as { detail?: string };
    return parsed.detail ?? err.message;
  } catch {
    return err.message || fallback;
  }
}

function formatTime(seconds: number) {
  const safe = Math.max(0, Math.round(seconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const rest = safe % 60;
  if (hours > 0) return `${hours}:${minutes.toString().padStart(2, "0")}:${rest.toString().padStart(2, "0")}`;
  return `${minutes}:${rest.toString().padStart(2, "0")}`;
}

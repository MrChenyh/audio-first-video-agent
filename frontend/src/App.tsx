import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock3,
  FileAudio,
  FileVideo,
  Gauge,
  Image as ImageIcon,
  Loader2,
  MessageSquareText,
  Play,
  SearchCheck,
  UploadCloud
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type JobStatus = "queued" | "running" | "succeeded" | "failed";

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
  };
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
  question: string;
  answer: {
    direct_answer: string;
    summary: string;
    evidence_refs: string[];
    uncertainties: string[];
  };
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

const nodeLabels: Record<string, string> = {
  queued: "排队中",
  starting: "启动任务",
  ingest_video: "读取视频",
  extract_audio: "抽取音频",
  transcribe_audio: "识别音频",
  build_audio_world_model: "构建音频先验",
  plan_keyframes: "规划目标帧",
  extract_keyframes: "抽取目标帧",
  observe_frames: "视觉验证",
  predict_next_events: "生成预测",
  verify_predictions: "验证预测",
  synthesize_answer: "生成总结",
  complete: "完成",
  failed: "失败"
};

export function App() {
  const [video, setVideo] = useState<File | null>(null);
  const [question, setQuestion] = useState("这个视频主要发生了什么？请按时间线总结，并给出关键证据。");
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const statusLabel = useMemo(() => {
    if (!job) return "等待上传";
    if (job.status === "failed") return "分析失败";
    if (job.status === "succeeded") return "分析完成";
    return nodeLabels[job.current_node] ?? job.current_node;
  }, [job]);

  useEffect(() => {
    return () => eventSourceRef.current?.close();
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!video) {
      setError("请选择一个视频文件。");
      return;
    }
    setSubmitting(true);
    setError(null);
    setResult(null);
    setJob(null);

    const body = new FormData();
    body.append("video", video);
    body.append("question", question);

    try {
      const response = await fetch("/api/jobs", { method: "POST", body });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as { job_id: string };
      subscribe(payload.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败。");
    } finally {
      setSubmitting(false);
    }
  }

  function subscribe(jobId: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(`/api/jobs/${jobId}/events`);
    eventSourceRef.current = source;
    source.onmessage = async (event) => {
      const nextJob = JSON.parse(event.data) as Job;
      setJob(nextJob);
      if (nextJob.status === "succeeded") {
        source.close();
        const response = await fetch(`/api/jobs/${nextJob.job_id}/result`);
        if (response.ok) setResult((await response.json()) as Result);
      }
      if (nextJob.status === "failed") {
        source.close();
        setError(nextJob.error ?? "任务失败。");
      }
    };
    source.onerror = () => {
      source.close();
      setError("进度连接中断，请刷新任务状态。");
    };
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
            <label className="drop-zone">
              <input
                type="file"
                accept="video/*"
                onChange={(event) => setVideo(event.target.files?.[0] ?? null)}
              />
              <UploadCloud size={28} />
              <span>{video ? video.name : "选择 1 到 10 分钟的视频"}</span>
            </label>

            <label className="question-box">
              <span>
                <MessageSquareText size={16} /> 问题
              </span>
              <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={5} />
            </label>

            <button className="primary-action" type="submit" disabled={submitting || !video}>
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
              {["transcribe_audio", "build_audio_world_model", "plan_keyframes", "observe_frames", "verify_predictions", "synthesize_answer"].map((node) => (
                <div className={job?.current_node === node ? "stage active" : "stage"} key={node}>
                  <Clock3 size={14} />
                  <span>{nodeLabels[node]}</span>
                </div>
              ))}
            </div>
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

        {result && <ResultView result={result} />}
      </section>
    </main>
  );
}

function ResultView({ result }: { result: Result }) {
  return (
    <section className="result-grid">
      <div className="answer-band">
        <div className="section-title">
          <CheckCircle2 size={18} />
          <h2>最终回答</h2>
        </div>
        <p className="direct-answer">{result.answer.direct_answer}</p>
        <p>{result.answer.summary}</p>
      </div>

      <div className="timeline-band">
        <div className="section-title">
          <Activity size={18} />
          <h2>音频先验</h2>
        </div>
        {result.transcription_status && (
          <div className="audio-status">
            <strong>{result.transcription_status.status ?? "unknown"} / {result.transcription_status.method ?? "unknown"}</strong>
            <span>{result.transcription_status.reason ?? "没有返回转写状态。"}</span>
            {result.transcription_status.local_error && <small>{result.transcription_status.local_error}</small>}
          </div>
        )}
        {Boolean(result.transcript_segments?.length) && (
          <div className="transcript-list">
            <div className="section-title compact">
              <FileAudio size={16} />
              <h3>音频转写</h3>
            </div>
            {result.transcript_segments?.map((segment, index) => (
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
                {event.visual_question && <small>{event.visual_question}</small>}
                {event.expected_visuals.length > 0 && <small>{event.expected_visuals.join(" / ")}</small>}
              </div>
            </article>
          ))}
        </div>
      </div>

      <div className="frames-band">
        <div className="section-title">
          <SearchCheck size={18} />
          <h2>目标帧验证</h2>
        </div>
        <div className="frame-grid">
          {result.frames.map((frame) => (
            <article className="frame-card" key={frame.filename}>
              <img src={frame.url} alt={`frame at ${formatTime(frame.time)}`} loading="lazy" />
              <div>
                <span className={`align ${frame.observation?.audio_alignment ?? "uncertain"}`}>
                  {alignmentLabel(frame.observation?.audio_alignment)}
                </span>
                <h3>{formatTime(frame.time)}</h3>
                <p className="target-text">{frame.observation?.visual_target ?? frame.probe?.question ?? frame.reason}</p>
                <p>{frame.observation?.scene ?? frame.reason}</p>
                {frame.observation?.evidence_assessment && <small>{frame.observation.evidence_assessment}</small>}
              </div>
            </article>
          ))}
        </div>
      </div>

      <div className="checks-band">
        <div className="section-title">
          <FileVideo size={18} />
          <h2>预测验证</h2>
        </div>
        <div className="check-list">
          {result.prediction_checks.map((check, index) => (
            <article className="check-row" key={`${check.window_start}-${index}`}>
              <span className={`align ${check.status}`}>{alignmentLabel(check.status)}</span>
              <div>
                <h3>{formatTime(check.window_start)} - {formatTime(check.window_end)}</h3>
                <p>{check.hypothesis}</p>
                {Boolean(check.expected_evidence?.length) && <small>{check.expected_evidence?.join(" / ")}</small>}
                <small>{check.evidence} / conflict {Math.round(check.conflict_score * 100)}%</small>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function alignmentLabel(status?: "match" | "conflict" | "uncertain") {
  if (status === "match") return "匹配";
  if (status === "conflict") return "冲突";
  return "待确认";
}

function formatTime(seconds: number) {
  const safe = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(safe / 60);
  const rest = safe % 60;
  return `${minutes}:${rest.toString().padStart(2, "0")}`;
}

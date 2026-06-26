import {
  Activity,
  AlertCircle,
  Globe2,
  Loader2,
  Play,
  Radio,
  Send,
  Square,
  UploadCloud,
  Video
} from "lucide-react";
import type { FormEvent, ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

type JobStatus = "queued" | "running" | "succeeded" | "failed";
type LiveStatus = "queued" | "running" | "streaming" | "stopping" | "stopped" | "succeeded" | "failed";
type ComposerMode = "video" | "live";

type Job = {
  job_id: string;
  question: string;
  status: JobStatus;
  progress: number;
  current_node: string;
  error?: string | null;
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
    sections?: Array<{ title: string; items: string[] }>;
  } | null;
  timeline: Array<{
    time: number;
    end_time?: number | null;
    label: string;
    evidence: string;
    expected_visuals: string[];
    visual_question?: string;
  }>;
  transcript_segments?: Array<{
    start: number;
    end: number;
    speaker?: string;
    text: string;
    confidence?: number | null;
    source?: string;
  }>;
  frames: Array<{
    time: number;
    filename: string;
    url: string;
    reason: string;
    observation?: {
      scene?: string;
      evidence_assessment?: string;
      notes?: string;
    } | null;
  }>;
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
    segment_count?: number;
  };
};

type LiveViolation = {
  source?: "audio" | "visual" | string;
  category: string;
  category_label?: string;
  severity?: "low" | "medium" | "high" | string;
  confidence?: number;
  evidence?: string;
  matched_text?: string;
  context?: string;
  visible_text?: string[];
};

type LiveSegment = {
  index: number;
  start_time: number;
  end_time: number;
  transcript?: string;
  summary: string;
  elapsed_seconds?: number;
  transcript_seconds?: number;
  vision_seconds?: number;
  analysis_seconds?: number;
  frame?: {
    time: number;
    filename: string;
    url: string;
    reason: string;
  };
  moderation?: {
    has_risk: boolean;
    risk_level: "none" | "low" | "medium" | "high" | string;
    violations: LiveViolation[];
    summary?: string;
  };
};

type LiveModel = {
  status: "warming_up" | "ready";
  program_type: string;
  current_focus: string;
  stable_summary: string;
  confidence: number;
  evidence_count: number;
  audio_evidence_count: number;
  visual_evidence_count: number;
  segment_count: number;
  risk_state?: {
    status?: "clear" | "alert" | string;
    scanned_segments?: number;
    risk_segments?: number;
    last_alert_time?: number | null;
    highest_level?: string;
    category_counts?: Record<string, number>;
    recent_alerts?: Array<{
      segment_index?: number;
      time?: number;
      risk_level?: string;
      categories?: string[];
      summary?: string;
      violations?: LiveViolation[];
    }>;
  };
};

type LiveSession = {
  session_id: string;
  source_url: string;
  question: string;
  status: LiveStatus;
  current_node: string;
  error?: string | null;
  resolved_url?: string | null;
  live_model?: LiveModel;
  segments: LiveSegment[];
};

type ChatMessage = {
  role: "user" | "assistant" | "system";
  text: string;
  meta?: string;
};

type PendingLive = {
  url: string;
};

type PendingVideoQuestion = {
  text: string;
  useWebSearch: boolean;
};

const PREPARE_VIDEO_QUESTION = "用户稍后会在对话中提问；请先解析视频内容，准备支持问答。";
const DEFAULT_LIVE_MONITOR = "实时监控直播是否出现违禁词、粗口、擦边、抽烟、暴力、危险行为等风险；只有命中风险时保留证据。";

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
  synthesize_answer: "生成回答",
  complete: "完成",
  failed: "失败"
};

export function App() {
  const [composerMode, setComposerMode] = useState<ComposerMode>("video");
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [liveSession, setLiveSession] = useState<LiveSession | null>(null);
  const [pendingLive, setPendingLive] = useState<PendingLive | null>(null);
  const [chat, setChat] = useState<ChatMessage[]>([
    {
      role: "assistant",
      text: "选择上传本地视频，或切到直播 URL。视频会先在后台解析，然后你可以连续提问；直播会先问你要监控什么，再开始实时扫描。"
    }
  ]);
  const [input, setInput] = useState("");
  const [pendingVideoQuestions, setPendingVideoQuestions] = useState<PendingVideoQuestion[]>([]);
  const [useWebSearch, setUseWebSearch] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [asking, setAsking] = useState(false);
  const [liveSubmitting, setLiveSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const liveEventSourceRef = useRef<EventSource | null>(null);
  const partialTimerRef = useRef<number | null>(null);
  const sourceVideoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const pendingVideoQuestionsRef = useRef<PendingVideoQuestion[]>([]);
  const useWebSearchRef = useRef(false);

  const sourceVideoUrl = job ? `/api/jobs/${job.job_id}/source` : null;
  const status = useMemo(() => statusText(job, liveSession, composerMode, pendingLive), [job, liveSession, composerMode, pendingLive]);
  const progress = job?.progress ?? (liveSession ? 100 : 0);
  const mediaTitle = liveSession ? "直播监控" : job ? "视频解析" : pendingLive ? "等待监控目标" : "等待媒体";
  const mediaSubtitle = liveSession?.source_url || pendingLive?.url || (job ? "视频正在后台解析，可先输入问题。" : "上方会播放视频；直播接入后显示实时状态。");
  const hasRunningTask = job?.status === "queued" || job?.status === "running" || liveSession?.status === "queued" || liveSession?.status === "running" || liveSession?.status === "streaming";

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
      liveEventSourceRef.current?.close();
      stopPartialPolling();
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chat, result?.answer?.direct_answer, liveSession?.live_model?.stable_summary, liveSession?.segments.length]);

  useEffect(() => {
    useWebSearchRef.current = useWebSearch;
  }, [useWebSearch]);

  function appendMessage(message: ChatMessage) {
    setChat((items) => [...items, message]);
  }

  function resetVideoState() {
    eventSourceRef.current?.close();
    stopPartialPolling();
    setJob(null);
    setResult(null);
    setError(null);
    clearPendingVideoQuestions();
  }

  function resetLiveState() {
    liveEventSourceRef.current?.close();
    setLiveSession(null);
    setPendingLive(null);
    setLiveError(null);
  }

  async function handleVideoSelected(file: File | null) {
    if (!file) return;
    setComposerMode("video");
    resetVideoState();
    resetLiveState();
    setChat([
      { role: "user", text: `上传本地视频：${file.name}` },
      { role: "assistant", text: "收到，我会先在后台解析音频和关键画面。你现在可以直接问这个视频的问题。" }
    ]);
    setSubmitting(true);
    try {
      const payload = await createFileJob(file, PREPARE_VIDEO_QUESTION);
      subscribeJob(payload.job_id);
    } catch (err) {
      const message = errorMessage(err, "上传失败。");
      setError(message);
      appendMessage({ role: "assistant", text: message });
    } finally {
      setSubmitting(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleComposerSubmit(event: FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text) return;
    setInput("");

    if (composerMode === "live") {
      await handleLiveComposer(text);
      return;
    }
    await handleVideoComposer(text);
  }

  async function handleVideoComposer(text: string) {
    if (!job && !result) {
      if (!isLikelyUrl(text)) {
        setChat((items) => [
          ...items,
          { role: "user", text },
          { role: "assistant", text: "先上传本地视频，或直接发一个视频网页/直链 URL；解析完成后我会按你的问题回答。" }
        ]);
        return;
      }
      await startVideoUrlJob(text);
      return;
    }
    if (job && job.status !== "succeeded") {
      enqueuePendingVideoQuestion(text);
      setChat((items) => [
        ...items,
        { role: "user", text },
        { role: "assistant", text: "我先记下这个问题。后台解析完成后，我会基于完整音频和关键画面来回答。" }
      ]);
      return;
    }
    await askVideoQuestion(text);
  }

  async function handleLiveComposer(text: string) {
    if (pendingLive) {
      await startLiveSession(pendingLive.url, text);
      return;
    }
    if (!liveSession) {
      if (!isLikelyUrl(text)) {
        setChat((items) => [
          ...items,
          { role: "user", text },
          { role: "assistant", text: "先发我直播 URL。接入后我会让你确认要监控什么。" }
        ]);
        return;
      }
      resetVideoState();
      setPendingLive({ url: text });
      setChat((items) => [
        ...items,
        { role: "user", text: `接入直播 URL：${text}` },
        { role: "assistant", text: "收到。你需要监控什么？例如：违禁词、粗口、擦边、抽烟、危险行为。" }
      ]);
      return;
    }
    setChat((items) => [
      ...items,
      { role: "user", text },
      { role: "assistant", text: `收到。当前直播已经按原目标运行；这条补充我先记录为新的关注点：${text}` }
    ]);
  }

  async function startVideoUrlJob(url: string) {
    const trimmed = url.trim();
    setComposerMode("video");
    resetVideoState();
    resetLiveState();
    setChat([
      { role: "user", text: `解析视频 URL：${trimmed}` },
      { role: "assistant", text: "收到，我会先下载并解析这个视频。解析过程中也可以直接继续提问。" }
    ]);
    setSubmitting(true);
    try {
      const response = await fetch("/api/jobs/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: trimmed, question: PREPARE_VIDEO_QUESTION })
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as { job_id: string };
      subscribeJob(payload.job_id);
    } catch (err) {
      const message = errorMessage(err, "URL 创建任务失败。");
      setError(message);
      appendMessage({ role: "assistant", text: message });
    } finally {
      setSubmitting(false);
    }
  }

  async function askVideoQuestion(question: string) {
    const jobId = job?.job_id ?? result?.job_id;
    if (!jobId) return;
    setChat((items) => [...items, { role: "user", text: question }]);
    await answerVideoQuestion(jobId, question, useWebSearchRef.current);
  }

  async function answerVideoQuestion(jobId: string, question: string, useWeb = useWebSearchRef.current) {
    setAsking(true);
    try {
      const response = await fetch(`/api/jobs/${jobId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, use_web_search: useWeb })
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as { answer: { answer: string; web_sources?: string[] } };
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: payload.answer.answer,
          meta: useWeb && payload.answer.web_sources?.length ? `联网参考：${payload.answer.web_sources.slice(0, 2).join("、")}` : undefined
        }
      ]);
    } catch (err) {
      setChat((items) => [...items, { role: "assistant", text: errorMessage(err, "追问失败。") }]);
    } finally {
      setAsking(false);
    }
  }

  function enqueuePendingVideoQuestion(question: string) {
    pendingVideoQuestionsRef.current = [...pendingVideoQuestionsRef.current, { text: question, useWebSearch: useWebSearchRef.current }];
    setPendingVideoQuestions(pendingVideoQuestionsRef.current);
  }

  function clearPendingVideoQuestions() {
    pendingVideoQuestionsRef.current = [];
    setPendingVideoQuestions([]);
  }

  async function startLiveSession(url: string, monitorQuestion: string) {
    const target = monitorQuestion.trim() || DEFAULT_LIVE_MONITOR;
    setComposerMode("live");
    resetVideoState();
    setPendingLive(null);
    setLiveSession(null);
    setLiveError(null);
    setChat((items) => [
      ...items,
      { role: "user", text: target },
      { role: "assistant", text: `开始监控。目标：${target}` }
    ]);
    setLiveSubmitting(true);
    try {
      const response = await fetch("/api/live/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, question: target, window_seconds: 2, max_segments: 0 })
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = (await response.json()) as { session_id: string };
      subscribeLive(payload.session_id);
    } catch (err) {
      const message = errorMessage(err, "直播监控启动失败。");
      setLiveError(message);
      appendMessage({ role: "assistant", text: message });
    } finally {
      setLiveSubmitting(false);
    }
  }

  async function stopLive() {
    if (!liveSession) return;
    await fetch(`/api/live/sessions/${liveSession.session_id}/stop`, { method: "POST" });
    liveEventSourceRef.current?.close();
    appendMessage({ role: "assistant", text: "已发送停止指令。" });
  }

  async function createFileJob(file: File, question: string) {
    const body = new FormData();
    body.append("video", file);
    body.append("question", question);
    const response = await fetch("/api/jobs", { method: "POST", body });
    if (!response.ok) throw new Error(await response.text());
    return (await response.json()) as { job_id: string };
  }

  function subscribeJob(jobId: string) {
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
        if (response.ok) {
          const payload = (await response.json()) as Result;
          setResult(payload);
          const queuedQuestions = [...pendingVideoQuestionsRef.current];
          clearPendingVideoQuestions();
          if (queuedQuestions.length > 0) {
            appendMessage({ role: "assistant", text: `后台解析完成。我现在回答刚才记下的 ${queuedQuestions.length} 个问题。` });
            for (const queuedQuestion of queuedQuestions) {
              await answerVideoQuestion(nextJob.job_id, queuedQuestion.text, queuedQuestion.useWebSearch);
            }
          } else {
            appendMessage({ role: "assistant", text: "后台解析完成。你可以继续问更具体的问题，我会尽量按你的问题来分析。" });
          }
        }
      }
      if (nextJob.status === "failed") {
        source.close();
        stopPartialPolling();
        const message = nextJob.error ?? "任务失败。";
        setError(message);
        appendMessage({ role: "assistant", text: message });
      }
    };
    source.onerror = () => {
      source.close();
      stopPartialPolling();
      const message = "进度连接中断，请刷新任务状态。";
      setError(message);
      appendMessage({ role: "assistant", text: message });
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
        // SSE owns fatal progress errors; partial polling only keeps context fresh.
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

  function subscribeLive(sessionId: string) {
    liveEventSourceRef.current?.close();
    const source = new EventSource(`/api/live/sessions/${sessionId}/events`);
    liveEventSourceRef.current = source;
    source.onmessage = (event) => {
      const payload = JSON.parse(event.data) as LiveSession;
      setLiveSession(payload);
      if (payload.status === "failed") {
        source.close();
        const message = payload.error ?? "直播监控失败。";
        setLiveError(message);
        appendMessage({ role: "assistant", text: message });
      }
      if (["stopped", "succeeded"].includes(payload.status)) {
        source.close();
      }
    };
    source.onerror = () => {
      source.close();
      const message = "直播事件连接中断，请重新启动直播监控。";
      setLiveError(message);
      appendMessage({ role: "assistant", text: message });
    };
  }

  function seekSourceVideo(seconds: number) {
    const player = sourceVideoRef.current;
    if (!player) return;
    const safeSeconds = Math.max(0, seconds);
    const jump = () => {
      player.currentTime = safeSeconds;
      player.scrollIntoView({ behavior: "smooth", block: "center" });
      void player.play().catch(() => undefined);
    };
    if (player.readyState === 0) {
      player.addEventListener("loadedmetadata", jump, { once: true });
      player.load();
      return;
    }
    jump();
  }

  return (
    <main className="app-shell">
      <section className="workspace-shell">
        <header className="topbar">
          <div>
            <p className="eyebrow">Audio-first multimodal agent</p>
            <h1>视频/直播问答工作台</h1>
          </div>
          <div className="status-pill" title="当前状态">
            {hasRunningTask || submitting || asking || liveSubmitting ? <Loader2 className="spin" size={16} /> : <Activity size={16} />}
            <span>{status}</span>
          </div>
        </header>

        <section className="player-panel">
          <div className="player-frame">
            {sourceVideoUrl ? (
              <video key={`${sourceVideoUrl}-${job?.status}`} ref={sourceVideoRef} className="source-video" src={sourceVideoUrl} controls preload="metadata" />
            ) : liveSession ? (
              <LiveStage session={liveSession} />
            ) : pendingLive ? (
              <EmptyStage icon={<Radio size={32} />} title="直播 URL 已收到" subtitle="请在下面输入要监控的目标，之后才会开始扫描。" />
            ) : (
              <EmptyStage icon={<Play size={32} />} title="等待视频或直播" subtitle="上传本地视频，或切到直播 URL 开始。" />
            )}
          </div>
          <div className="media-meta">
            <div>
              <strong>{mediaTitle}</strong>
              <span>{mediaSubtitle}</span>
            </div>
            <div className="mini-progress" aria-label="分析进度">
              <span>{job ? `${job.progress}%` : liveSession ? liveStatusLabel(liveSession.status) : pendingLive ? "待确认" : "待开始"}</span>
              <div><i style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} /></div>
            </div>
          </div>
          {(error || liveError) && (
            <div className="error-line" role="alert">
              <AlertCircle size={16} />
              <span>{error || liveError}</span>
            </div>
          )}
        </section>

        <section className="conversation-panel">
          <div className="chat-list" aria-live="polite">
            {chat.map((message, index) => (
              <article className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
                <p>{renderTimeLinkedText(message.text, seekSourceVideo)}</p>
                {message.meta && <small>{message.meta}</small>}
              </article>
            ))}
            {liveSession?.live_model && <LiveModelMessage session={liveSession} />}
            {(asking || submitting || liveSubmitting) && <ThinkingMessage label={thinkingLabel(asking, submitting, liveSubmitting, useWebSearch)} />}
            <div ref={chatEndRef} />
          </div>

          <form className="composer" onSubmit={handleComposerSubmit}>
            <input ref={fileInputRef} type="file" accept="video/*" hidden onChange={(event) => void handleVideoSelected(event.target.files?.[0] ?? null)} />
            <div className="composer-tools" role="tablist" aria-label="输入来源">
              <button
                type="button"
                className={composerMode === "video" ? "active" : ""}
                onClick={() => {
                  setComposerMode("video");
                  setPendingLive(null);
                }}
              >
                <Video size={16} /> 视频问答
              </button>
              <button
                type="button"
                className={composerMode === "live" ? "active" : ""}
                onClick={() => {
                  setComposerMode("live");
                  resetVideoState();
                }}
              >
                <Radio size={16} /> 直播监控
              </button>
              <button type="button" onClick={() => fileInputRef.current?.click()} disabled={submitting || liveSubmitting}>
                <UploadCloud size={16} /> 上传本地视频
              </button>
              <button
                type="button"
                className={`web-tool ${useWebSearch ? "active" : ""}`}
                onClick={() => setUseWebSearch((value) => !value)}
                aria-pressed={useWebSearch}
                title="回答视频问题时结合联网搜索"
              >
                <Globe2 size={16} /> {useWebSearch ? "联网增强" : "仅视频库"}
              </button>
              {liveSession && ["queued", "running", "streaming"].includes(liveSession.status) && (
                <button type="button" className="danger-tool" onClick={() => void stopLive()}>
                  <Square size={14} /> 停止直播
                </button>
              )}
            </div>
            <div className="composer-row">
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder={composerPlaceholder(composerMode, Boolean(job || result), Boolean(pendingLive), Boolean(liveSession), pendingVideoQuestions.length)}
              />
              <button type="submit" disabled={asking || submitting || liveSubmitting || !input.trim()} aria-label="发送">
                {asking || submitting || liveSubmitting ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              </button>
            </div>
          </form>
        </section>
      </section>
    </main>
  );
}

function EmptyStage({ icon, title, subtitle }: { icon: ReactNode; title: string; subtitle: string }) {
  return (
    <div className="empty-stage">
      {icon}
      <strong>{title}</strong>
      <span>{subtitle}</span>
    </div>
  );
}

function ThinkingMessage({ label }: { label: string }) {
  return (
    <article className="chat-message assistant thinking-message" aria-live="polite">
      <span>{label}</span>
      <i />
      <i />
      <i />
    </article>
  );
}

function LiveStage({ session }: { session: LiveSession }) {
  const model = session.live_model;
  const riskState = model?.risk_state;
  const hasRisk = Boolean((riskState?.risk_segments ?? 0) > 0);
  const latestFrame = session.segments[session.segments.length - 1]?.frame;
  return (
    <div className={`live-stage ${hasRisk ? "alert" : "clear"}`}>
      {latestFrame?.url ? <img src={latestFrame.url} alt="直播风险截图" /> : <Radio size={34} />}
      <div>
        <strong>{hasRisk ? "发现风险" : liveStatusLabel(session.status)}</strong>
        <span>{model?.stable_summary || "正在接入直播并扫描风险。"}</span>
      </div>
    </div>
  );
}

function LiveModelMessage({ session }: { session: LiveSession }) {
  const model = session.live_model;
  if (!model) return null;
  const riskState = model.risk_state;
  const recentAlerts = riskState?.recent_alerts ?? [];
  return (
    <article className="chat-message assistant live-summary">
      <p>{model.stable_summary || model.current_focus}</p>
      <small>
        已扫描 {riskState?.scanned_segments ?? model.segment_count} 个窗口；告警 {riskState?.risk_segments ?? 0} 次；最高风险 {riskLevelLabel(riskState?.highest_level)}。
      </small>
      {recentAlerts.slice(-3).map((alert, index) => (
        <small key={`${alert.segment_index}-${index}`}>
          {formatTime(alert.time ?? 0)}：{alert.summary || "风险告警"}
        </small>
      ))}
    </article>
  );
}

function statusText(job: Job | null, liveSession: LiveSession | null, mode: ComposerMode, pendingLive: PendingLive | null) {
  if (liveSession) return liveStatusLabel(liveSession.status);
  if (pendingLive) return "等待监控目标";
  if (job?.status === "failed") return "分析失败";
  if (job?.status === "succeeded") return "分析完成";
  if (job) return labelForNode(job.current_node);
  return mode === "live" ? "等待直播 URL" : "等待视频";
}

function labelForNode(node: string) {
  if (node.startsWith("observe_frames")) return `观察关键画面 ${node.replace("observe_frames", "").trim()}`.trim();
  return nodeLabels[node] ?? node;
}

function composerPlaceholder(mode: ComposerMode, hasVideoContext: boolean, hasPendingLive: boolean, hasLiveSession: boolean, pendingQuestionCount: number) {
  if (mode === "live") {
    if (hasPendingLive) return "输入要监控的内容，例如：粗口、擦边、抽烟、危险动作";
    if (hasLiveSession) return "补充新的直播关注点";
    return "粘贴直播 URL";
  }
  if (pendingQuestionCount > 0) return `已记下 ${pendingQuestionCount} 个问题，还可以继续补充`;
  if (hasVideoContext) return "问这个视频一个问题，例如：详细总结一下升级点";
  return "粘贴视频 URL，或点击上传本地视频";
}

function isLikelyUrl(text: string) {
  return /^https?:\/\//i.test(text.trim());
}

function liveStatusLabel(status: LiveStatus) {
  if (status === "queued") return "排队中";
  if (status === "running") return "运行中";
  if (status === "streaming") return "监控中";
  if (status === "stopping") return "停止中";
  if (status === "stopped") return "已停止";
  if (status === "succeeded") return "已完成";
  return "失败";
}

function riskLevelLabel(level?: string) {
  if (level === "high") return "高";
  if (level === "medium") return "中";
  if (level === "low") return "低";
  return "无";
}

function thinkingLabel(asking: boolean, submitting: boolean, liveSubmitting: boolean, useWebSearch: boolean) {
  if (asking) return useWebSearch ? "正在结合视频知识库和联网资料思考" : "正在基于视频知识库思考";
  if (liveSubmitting) return "正在接入直播监控";
  if (submitting) return "正在创建解析任务";
  return "正在处理";
}

function renderTimeLinkedText(text: string, onSeek: (seconds: number) => void): ReactNode[] {
  const parts: ReactNode[] = [];
  const pattern = /(^|[^\d])(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    const prefix = match[1];
    const label = match[2];
    const labelIndex = match.index + prefix.length;
    if (labelIndex > lastIndex) parts.push(text.slice(lastIndex, labelIndex));
    parts.push(
      <button className="time-link" type="button" onClick={() => onSeek(secondsFromTimeLabel(label))} key={`${label}-${labelIndex}`}>
        {label}
      </button>
    );
    lastIndex = labelIndex + label.length;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts.length ? parts : [text];
}

function secondsFromTimeLabel(label: string) {
  const parts = label.split(":").map((part) => Number(part));
  if (parts.some((part) => Number.isNaN(part))) return 0;
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  return parts[0] * 60 + parts[1];
}

function formatTime(seconds: number) {
  const safe = Math.max(0, Math.round(seconds));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const rest = safe % 60;
  if (hours > 0) return `${hours}:${minutes.toString().padStart(2, "0")}:${rest.toString().padStart(2, "0")}`;
  return `${minutes}:${rest.toString().padStart(2, "0")}`;
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

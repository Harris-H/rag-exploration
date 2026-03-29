'use client';

import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import SearchInput from '@/components/SearchInput';
import PipelineStep from '@/components/PipelineStep';
import StreamingText from '@/components/StreamingText';
import ResultCard from '@/components/ResultCard';
import { ragQueryStream, type EmbeddingResult } from '@/lib/api';

type StepStatus = 'idle' | 'active' | 'done';

interface PipelineState {
  input: StepStatus;
  retrieval: StepStatus;
  prompt: StepStatus;
  generation: StepStatus;
}

export default function RouteCPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<PipelineState>({
    input: 'idle', retrieval: 'idle', prompt: 'idle', generation: 'idle',
  });
  const [retrievalResults, setRetrievalResults] = useState<EmbeddingResult[]>([]);
  const [promptText, setPromptText] = useState('');
  const [ragAnswer, setRagAnswer] = useState('');
  const [directAnswer, setDirectAnswer] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setError(null);
    setRetrievalResults([]);
    setPromptText('');
    setRagAnswer('');
    setDirectAnswer('');
    setIsStreaming(false);
    setElapsedMs(null);
    setPipeline({ input: 'idle', retrieval: 'idle', prompt: 'idle', generation: 'idle' });
  }, []);

  const handleSearch = useCallback((query: string) => {
    // Abort previous request
    if (controllerRef.current) {
      controllerRef.current.abort();
    }

    reset();
    setLoading(true);
    setPipeline({ input: 'done', retrieval: 'active', prompt: 'idle', generation: 'idle' });

    const controller = ragQueryStream(
      query,
      3,
      // onEvent
      (event) => {
        const { type, data } = event;

        if (type === 'retrieval') {
          const d = data as { results?: EmbeddingResult[] };
          setRetrievalResults(d.results || []);
          setPipeline((p) => ({ ...p, retrieval: 'done', prompt: 'active' }));
        } else if (type === 'prompt') {
          const d = data as { prompt?: string };
          setPromptText(d.prompt || '');
          setPipeline((p) => ({ ...p, prompt: 'done', generation: 'active' }));
          setIsStreaming(true);
        } else if (type === 'token') {
          const d = data as { token?: string };
          setRagAnswer((prev) => prev + (d.token || ''));
        } else if (type === 'done') {
          const d = data as { direct_answer?: string; elapsed_ms?: number };
          setDirectAnswer(d.direct_answer || '');
          setElapsedMs(d.elapsed_ms ?? null);
          setPipeline({ input: 'done', retrieval: 'done', prompt: 'done', generation: 'done' });
          setIsStreaming(false);
          setLoading(false);
        }
      },
      // onError
      (err) => {
        setError(err.message || '请求失败，请确认后端服务是否运行');
        setLoading(false);
        setIsStreaming(false);
        setPipeline({ input: 'done', retrieval: 'idle', prompt: 'idle', generation: 'idle' });
      },
      // onDone
      () => {
        setIsStreaming(false);
        setLoading(false);
        setPipeline((p) => ({ ...p, generation: 'done' }));
      },
    );

    controllerRef.current = controller;
  }, [reset]);

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-xs font-mono px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-400">
            路线 C
          </span>
          <h1 className="text-2xl font-bold">完整 RAG 管道</h1>
        </div>
        <p className="text-white/40 text-sm leading-relaxed max-w-2xl">
          完整的检索增强生成流程：用户输入 → 文档检索 → Prompt 构建 → LLM 流式生成。
          使用本地 Ollama 模型，生成可能需要 30-120 秒，请耐心等待。
        </p>
      </motion.div>

      {/* Search */}
      <div className="mb-10">
        <SearchInput onSearch={handleSearch} loading={loading} placeholder="输入你的问题..." />
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mb-8 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm"
          >
            ⚠️ {error}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Pipeline visualization */}
        <div className="lg:col-span-1">
          <div className="sticky top-20">
            <h3 className="text-xs font-medium text-white/40 mb-4">处理流程</h3>
            <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.06]">
              <PipelineStep number={1} title="用户输入" description="接收查询内容" status={pipeline.input} />
              <PipelineStep number={2} title="文档检索" description="从知识库检索相关文档" status={pipeline.retrieval} />
              <PipelineStep number={3} title="Prompt 构建" description="将检索结果注入提示词" status={pipeline.prompt} />
              <PipelineStep number={4} title="LLM 生成" description="大语言模型生成答案" status={pipeline.generation} isLast />
            </div>

            {elapsedMs !== null && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-3 text-xs text-white/30 text-center"
              >
                总耗时: <strong className="text-white/60">{(elapsedMs / 1000).toFixed(1)}s</strong>
              </motion.div>
            )}
          </div>
        </div>

        {/* Right: Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Retrieval results */}
          <AnimatePresence>
            {retrievalResults.length > 0 && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <h3 className="text-xs font-medium text-white/40 mb-3">📄 检索到的文档</h3>
                <div className="space-y-2">
                  {retrievalResults.map((r, i) => (
                    <ResultCard key={i} title={r.title} score={r.score} content={r.content} index={i} scoreLabel="相关度" />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Prompt preview */}
          <AnimatePresence>
            {promptText && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <h3 className="text-xs font-medium text-white/40 mb-3">📝 构建的 Prompt</h3>
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.06] max-h-48 overflow-auto">
                  <pre className="text-xs text-white/40 whitespace-pre-wrap break-words font-mono leading-relaxed">
                    {promptText}
                  </pre>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* RAG Answer */}
          <AnimatePresence>
            {(ragAnswer || isStreaming) && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <h3 className="text-xs font-medium text-emerald-400 mb-3">🤖 RAG 回答</h3>
                <div className="p-5 rounded-xl bg-emerald-500/5 border border-emerald-500/15">
                  <StreamingText text={ragAnswer} isStreaming={isStreaming} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Comparison: RAG vs Direct */}
          <AnimatePresence>
            {directAnswer && ragAnswer && !isStreaming && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <h3 className="text-xs font-medium text-white/40 mb-3">🔀 对比：RAG 回答 vs 直接回答</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/15">
                    <div className="text-xs font-medium text-emerald-400 mb-2 flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      RAG 回答（基于检索）
                    </div>
                    <p className="text-white/70 text-sm leading-relaxed">{ragAnswer}</p>
                  </div>
                  <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/15">
                    <div className="text-xs font-medium text-amber-400 mb-2 flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                      直接回答（无检索）
                    </div>
                    <p className="text-white/70 text-sm leading-relaxed">{directAnswer}</p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Empty state */}
          {!loading && !ragAnswer && !error && retrievalResults.length === 0 && (
            <div className="text-center py-16 text-white/20 text-sm">
              输入问题开始体验完整的 RAG 流程
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

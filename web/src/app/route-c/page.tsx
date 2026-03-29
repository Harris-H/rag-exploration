'use client';

import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import SearchInput from '@/components/SearchInput';
import PipelineStep from '@/components/PipelineStep';
import StreamingText from '@/components/StreamingText';
import ResultCard from '@/components/ResultCard';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
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
          // data is JSON array of results
          const results = Array.isArray(data) ? data as EmbeddingResult[] : [];
          setRetrievalResults(results);
          setPipeline((p) => ({ ...p, retrieval: 'done', prompt: 'active' }));
        } else if (type === 'prompt') {
          // data is plain text string
          const promptStr = typeof data === 'string' ? data : '';
          setPromptText(promptStr);
          setPipeline((p) => ({ ...p, prompt: 'done', generation: 'active' }));
          setIsStreaming(true);
        } else if (type === 'token') {
          // data is a single token string
          const token = typeof data === 'string' ? data : '';
          setRagAnswer((prev) => prev + token);
        } else if (type === 'done') {
          // data is JSON {full_answer, retrieval_count, model}
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
          <Badge variant="outline" className="font-mono bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/30">
            路线 C
          </Badge>
          <h1 className="text-2xl font-bold text-foreground">完整 RAG 管道</h1>
        </div>
        <p className="text-muted-foreground text-sm leading-relaxed max-w-2xl">
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
            className="mb-8 p-4 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-sm"
          >
            ⚠️ {error}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Pipeline visualization */}
        <div className="lg:col-span-1">
          <div className="sticky top-20">
            <h3 className="text-xs font-medium text-muted-foreground mb-4">处理流程</h3>
            <Card>
              <CardContent>
                <PipelineStep number={1} title="用户输入" description="接收查询内容" status={pipeline.input} />
                <PipelineStep number={2} title="文档检索" description="从知识库检索相关文档" status={pipeline.retrieval} />
                <PipelineStep number={3} title="Prompt 构建" description="将检索结果注入提示词" status={pipeline.prompt} />
                <PipelineStep number={4} title="LLM 生成" description="大语言模型生成答案" status={pipeline.generation} isLast />
              </CardContent>
            </Card>

            {elapsedMs !== null && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-3 text-xs text-muted-foreground text-center"
              >
                总耗时: <strong className="text-foreground">{(elapsedMs / 1000).toFixed(1)}s</strong>
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
                <h3 className="text-xs font-medium text-muted-foreground mb-3">📄 检索到的文档</h3>
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
                <h3 className="text-xs font-medium text-muted-foreground mb-3">📝 构建的 Prompt</h3>
                <Card>
                  <CardContent className="max-h-48 overflow-auto">
                    <pre className="text-xs text-muted-foreground whitespace-pre-wrap break-words font-mono leading-relaxed">
                      {promptText}
                    </pre>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>

          {/* RAG Answer */}
          <AnimatePresence>
            {(ragAnswer || isStreaming) && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <h3 className="text-xs font-medium text-emerald-600 dark:text-emerald-400 mb-3">🤖 RAG 回答</h3>
                <div className="p-5 rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/30">
                  <StreamingText text={ragAnswer} isStreaming={isStreaming} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Empty state */}
          {!loading && !ragAnswer && !error && retrievalResults.length === 0 && (
            <div className="text-center py-16 text-muted-foreground text-sm">
              输入问题开始体验完整的 RAG 流程
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

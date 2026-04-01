'use client';

import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import SearchInput from '@/components/SearchInput';
import PipelineStep from '@/components/PipelineStep';
import StreamingText from '@/components/StreamingText';
import CitedText from '@/components/CitedText';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  enhancedRagQueryStream,
  ragQueryStream,
  type EnhancedChunkResult,
  type EnhancedRerankData,
  type ChunkingEventData,
  type EnhancedDoneData,
  type EmbeddingResult,
  type QueryExpansionData,
  type CitationSource,
} from '@/lib/api';
import { Settings2, Zap, ZapOff, ChevronDown, ChevronUp } from 'lucide-react';

type StepStatus = 'idle' | 'active' | 'done';

interface PipelineState {
  input: StepStatus;
  expansion: StepStatus;
  chunking: StepStatus;
  retrieval: StepStatus;
  reranking: StepStatus;
  prompt: StepStatus;
  generation: StepStatus;
}

const INITIAL_PIPELINE: PipelineState = {
  input: 'idle', expansion: 'idle', chunking: 'idle', retrieval: 'idle',
  reranking: 'idle', prompt: 'idle', generation: 'idle',
};

type Mode = 'enhanced' | 'baseline';

export default function RouteCPage() {
  const [mode, setMode] = useState<Mode>('enhanced');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<PipelineState>(INITIAL_PIPELINE);

  // Config
  const [useExpansion, setUseExpansion] = useState(false);
  const [useChunking, setUseChunking] = useState(true);
  const [chunkStrategy, setChunkStrategy] = useState('recursive');
  const [chunkSize, setChunkSize] = useState(200);
  const [useHybrid, setUseHybrid] = useState(true);
  const [useReranking, setUseReranking] = useState(true);
  const [showConfig, setShowConfig] = useState(true);

  // Results
  const [expansionData, setExpansionData] = useState<QueryExpansionData | null>(null);
  const [chunkingInfo, setChunkingInfo] = useState<ChunkingEventData | null>(null);
  const [retrievalResults, setRetrievalResults] = useState<(EnhancedChunkResult | EmbeddingResult)[]>([]);
  const [rerankData, setRerankData] = useState<EnhancedRerankData | null>(null);
  const [promptText, setPromptText] = useState('');
  const [ragAnswer, setRagAnswer] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [doneData, setDoneData] = useState<EnhancedDoneData | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setError(null);
    setExpansionData(null);
    setChunkingInfo(null);
    setRetrievalResults([]);
    setRerankData(null);
    setPromptText('');
    setRagAnswer('');
    setIsStreaming(false);
    setDoneData(null);
    setPipeline(INITIAL_PIPELINE);
  }, []);

  const handleSearch = useCallback((query: string) => {
    if (controllerRef.current) controllerRef.current.abort();
    reset();
    setLoading(true);

    if (mode === 'enhanced') {
      const skipChunking = !useChunking;
      const skipReranking = !useReranking;
      const skipExpansion = !useExpansion;
      setPipeline({
        input: 'done',
        expansion: skipExpansion ? 'done' : 'active',
        chunking: skipExpansion ? (skipChunking ? 'done' : 'active') : 'idle',
        retrieval: 'idle',
        reranking: 'idle',
        prompt: 'idle',
        generation: 'idle',
      });

      const controller = enhancedRagQueryStream(
        query,
        { topK: 3, useExpansion, useChunking, chunkStrategy, chunkSize, useHybrid, useReranking },
        (event) => {
          const { type, data } = event;
          if (type === 'config') {
            // config echo - skip
          } else if (type === 'expansion') {
            setExpansionData(data as QueryExpansionData);
            setPipeline(p => ({ ...p, expansion: 'done', chunking: skipChunking ? 'done' : 'active' }));
          } else if (type === 'chunking') {
            setChunkingInfo(data as ChunkingEventData);
            setPipeline(p => ({ ...p, expansion: 'done', chunking: 'done', retrieval: 'active' }));
          } else if (type === 'retrieval') {
            setRetrievalResults(data as EnhancedChunkResult[]);
            setPipeline(p => ({
              ...p,
              expansion: 'done',
              chunking: 'done',
              retrieval: 'done',
              reranking: skipReranking ? 'done' : 'active',
            }));
          } else if (type === 'reranking') {
            setRerankData(data as EnhancedRerankData);
            setPipeline(p => ({ ...p, reranking: 'done', prompt: 'active' }));
          } else if (type === 'prompt') {
            setPromptText(typeof data === 'string' ? data : '');
            setPipeline(p => ({ ...p, reranking: 'done', prompt: 'done', generation: 'active' }));
            setIsStreaming(true);
          } else if (type === 'token') {
            setRagAnswer(prev => prev + (typeof data === 'string' ? data : ''));
          } else if (type === 'done') {
            const d = data as EnhancedDoneData;
            setDoneData(d);
            setPipeline({
              input: 'done', expansion: 'done', chunking: 'done', retrieval: 'done',
              reranking: 'done', prompt: 'done', generation: 'done',
            });
            setIsStreaming(false);
            setLoading(false);
          }
        },
        (err) => {
          setError(err.message || '请求失败');
          setLoading(false);
          setIsStreaming(false);
        },
        () => {
          setIsStreaming(false);
          setLoading(false);
        },
      );
      controllerRef.current = controller;
    } else {
      // Baseline mode: use original RAG
      setPipeline({
        input: 'done', expansion: 'done', chunking: 'done', retrieval: 'active',
        reranking: 'done', prompt: 'idle', generation: 'idle',
      });

      const controller = ragQueryStream(
        query, 3,
        (event) => {
          const { type, data } = event;
          if (type === 'retrieval') {
            setRetrievalResults(Array.isArray(data) ? data as EmbeddingResult[] : []);
            setPipeline(p => ({ ...p, retrieval: 'done', prompt: 'active' }));
          } else if (type === 'prompt') {
            setPromptText(typeof data === 'string' ? data : '');
            setPipeline(p => ({ ...p, prompt: 'done', generation: 'active' }));
            setIsStreaming(true);
          } else if (type === 'token') {
            setRagAnswer(prev => prev + (typeof data === 'string' ? data : ''));
          } else if (type === 'done') {
            setPipeline({
              input: 'done', expansion: 'done', chunking: 'done', retrieval: 'done',
              reranking: 'done', prompt: 'done', generation: 'done',
            });
            setIsStreaming(false);
            setLoading(false);
          }
        },
        (err) => {
          setError(err.message || '请求失败');
          setLoading(false);
          setIsStreaming(false);
        },
        () => { setIsStreaming(false); setLoading(false); },
      );
      controllerRef.current = controller;
    }
  }, [reset, mode, useExpansion, useChunking, chunkStrategy, chunkSize, useHybrid, useReranking]);

  const isEnhanced = mode === 'enhanced';

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <div className="flex items-center gap-3 mb-3">
          <Badge variant="outline" className="font-mono bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/30">
            路线 C
          </Badge>
          <h1 className="text-2xl font-bold text-foreground">完整 RAG 管道</h1>
          <div className="ml-auto flex items-center gap-2">
            <Button
              variant={isEnhanced ? 'default' : 'outline'}
              size="sm"
              onClick={() => { setMode('enhanced'); reset(); }}
              className="gap-1.5"
            >
              <Zap className="w-3.5 h-3.5" /> 增强模式
            </Button>
            <Button
              variant={!isEnhanced ? 'default' : 'outline'}
              size="sm"
              onClick={() => { setMode('baseline'); reset(); }}
              className="gap-1.5"
            >
              <ZapOff className="w-3.5 h-3.5" /> 基线模式
            </Button>
          </div>
        </div>
        <p className="text-muted-foreground text-sm leading-relaxed max-w-3xl">
          {isEnhanced
            ? '增强 RAG 管道：查询扩展 → 文档分块 → 混合检索(BM25+Embedding) → Cross-Encoder 重排序 → Prompt 构建 → LLM 流式生成。'
            : '基线 RAG 管道：Embedding 检索全文档 → Prompt 构建 → LLM 流式生成。切换到增强模式查看分块+重排序的效果。'}
        </p>
      </motion.div>

      {/* Config Panel (enhanced mode only) */}
      <AnimatePresence>
        {isEnhanced && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-6 overflow-hidden"
          >
            <Card className="border-emerald-200/50 dark:border-emerald-500/20">
              <CardContent className="pt-4 pb-3">
                <button
                  onClick={() => setShowConfig(!showConfig)}
                  className="flex items-center gap-2 w-full text-left text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                >
                  <Settings2 className="w-4 h-4" />
                  Pipeline 配置
                  {showConfig ? <ChevronUp className="w-4 h-4 ml-auto" /> : <ChevronDown className="w-4 h-4 ml-auto" />}
                </button>
                <AnimatePresence>
                  {showConfig && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4"
                    >
                      {/* Toggle: Query Expansion */}
                      <div className="space-y-2">
                        <label className="text-xs font-medium text-muted-foreground">查询扩展</label>
                        <button
                          onClick={() => setUseExpansion(!useExpansion)}
                          className={`w-full px-3 py-2 rounded-lg text-sm font-medium border transition-colors ${
                            useExpansion
                              ? 'bg-amber-50 dark:bg-amber-500/10 border-amber-300 dark:border-amber-500/30 text-amber-700 dark:text-amber-300'
                              : 'bg-muted/50 border-border text-muted-foreground'
                          }`}
                        >
                          {useExpansion ? '🔀 启用扩展' : '❌ 关闭'}
                        </button>
                      </div>

                      {/* Toggle: Chunking */}
                      <div className="space-y-2">
                        <label className="text-xs font-medium text-muted-foreground">文档分块</label>
                        <button
                          onClick={() => setUseChunking(!useChunking)}
                          className={`w-full px-3 py-2 rounded-lg text-sm font-medium border transition-colors ${
                            useChunking
                              ? 'bg-emerald-50 dark:bg-emerald-500/10 border-emerald-300 dark:border-emerald-500/30 text-emerald-700 dark:text-emerald-300'
                              : 'bg-muted/50 border-border text-muted-foreground'
                          }`}
                        >
                          {useChunking ? '✅ 启用分块' : '❌ 关闭（全文档）'}
                        </button>
                      </div>

                      {/* Strategy selector */}
                      <div className="space-y-2">
                        <label className="text-xs font-medium text-muted-foreground">分块策略</label>
                        <select
                          value={chunkStrategy}
                          onChange={(e) => setChunkStrategy(e.target.value)}
                          disabled={!useChunking}
                          className="w-full px-3 py-2 rounded-lg text-sm border border-border bg-background disabled:opacity-40"
                        >
                          <option value="recursive">递归分块 (推荐)</option>
                          <option value="sentence">句子边界</option>
                          <option value="sliding">滑动窗口</option>
                          <option value="fixed">固定长度</option>
                        </select>
                      </div>

                      {/* Chunk size */}
                      <div className="space-y-2">
                        <label className="text-xs font-medium text-muted-foreground">
                          分块大小: <span className="text-foreground">{chunkSize}</span> 字
                        </label>
                        <input
                          type="range"
                          min={50}
                          max={500}
                          step={25}
                          value={chunkSize}
                          onChange={(e) => setChunkSize(Number(e.target.value))}
                          disabled={!useChunking}
                          className="w-full disabled:opacity-40"
                        />
                      </div>

                      {/* Toggles row */}
                      <div className="space-y-2">
                        <label className="text-xs font-medium text-muted-foreground">检索优化</label>
                        <div className="flex gap-2">
                          <button
                            onClick={() => setUseHybrid(!useHybrid)}
                            className={`flex-1 px-2 py-2 rounded-lg text-xs font-medium border transition-colors ${
                              useHybrid
                                ? 'bg-blue-50 dark:bg-blue-500/10 border-blue-300 dark:border-blue-500/30 text-blue-700 dark:text-blue-300'
                                : 'bg-muted/50 border-border text-muted-foreground'
                            }`}
                          >
                            {useHybrid ? '混合' : '纯向量'}
                          </button>
                          <button
                            onClick={() => setUseReranking(!useReranking)}
                            className={`flex-1 px-2 py-2 rounded-lg text-xs font-medium border transition-colors ${
                              useReranking
                                ? 'bg-purple-50 dark:bg-purple-500/10 border-purple-300 dark:border-purple-500/30 text-purple-700 dark:text-purple-300'
                                : 'bg-muted/50 border-border text-muted-foreground'
                            }`}
                          >
                            {useReranking ? '重排序' : '无重排'}
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Search */}
      <div className="mb-8">
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
            <h3 className="text-xs font-medium text-muted-foreground mb-4">
              {isEnhanced ? '增强处理流程' : '基线处理流程'}
            </h3>
            <Card>
              <CardContent>
                <PipelineStep number={1} title="用户输入" description="接收查询内容" status={pipeline.input} />
                {isEnhanced && (
                  <PipelineStep
                    number={2}
                    title="查询扩展"
                    description={useExpansion ? 'LLM 生成查询变体' : '跳过'}
                    status={pipeline.expansion}
                  />
                )}
                {isEnhanced && (
                  <PipelineStep
                    number={3}
                    title="文档分块"
                    description={useChunking ? `${chunkStrategy} · ${chunkSize}字` : '跳过（全文档模式）'}
                    status={pipeline.chunking}
                  />
                )}
                <PipelineStep
                  number={isEnhanced ? 4 : 2}
                  title={isEnhanced && useHybrid ? '混合检索' : '文档检索'}
                  description={isEnhanced && useHybrid
                    ? (useExpansion ? '多查询 RRF 融合' : 'BM25 + Embedding + RRF')
                    : 'Embedding 向量相似度'}
                  status={pipeline.retrieval}
                />
                {isEnhanced && (
                  <PipelineStep
                    number={5}
                    title="重排序"
                    description={useReranking ? 'Cross-Encoder 精排' : '跳过'}
                    status={pipeline.reranking}
                  />
                )}
                <PipelineStep
                  number={isEnhanced ? 6 : 3}
                  title="Prompt 构建"
                  description="将检索结果注入提示词"
                  status={pipeline.prompt}
                />
                <PipelineStep
                  number={isEnhanced ? 7 : 4}
                  title="LLM 生成"
                  description="大语言模型生成答案"
                  status={pipeline.generation}
                  isLast
                />
              </CardContent>
            </Card>

            {doneData && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-3 text-xs text-muted-foreground text-center">
                总耗时: <strong className="text-foreground">{(doneData.elapsed_ms / 1000).toFixed(1)}s</strong>
                {' · '}{doneData.model}
              </motion.div>
            )}
          </div>
        </div>

        {/* Right: Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Query Expansion info */}
          <AnimatePresence>
            {expansionData && isEnhanced && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <h3 className="text-xs font-medium text-amber-600 dark:text-amber-400 mb-3">🔀 查询扩展</h3>
                <Card className="border-amber-200/50 dark:border-amber-500/20">
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 text-sm">
                        <Badge variant="outline" className="bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-500/30">
                          原始查询
                        </Badge>
                        <span className="text-foreground font-medium">{expansionData.original}</span>
                      </div>
                      {expansionData.variants.length > 0 ? (
                        <div className="space-y-1.5">
                          <span className="text-xs text-muted-foreground">LLM 生成的查询变体（共 {expansionData.total_queries} 个查询参与检索）：</span>
                          {expansionData.variants.map((v, i) => (
                            <div key={i} className="flex items-center gap-2 text-sm pl-2 border-l-2 border-amber-200 dark:border-amber-500/30">
                              <span className="text-xs text-muted-foreground font-mono">#{i + 1}</span>
                              <span className="text-foreground">{v}</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <span className="text-xs text-muted-foreground">未生成有效变体，使用原始查询检索</span>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Chunking info */}
          <AnimatePresence>
            {chunkingInfo && isEnhanced && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <h3 className="text-xs font-medium text-muted-foreground mb-3">✂️ 分块结果</h3>
                <Card>
                  <CardContent>
                    <div className="flex items-center gap-4 text-sm">
                      <Badge variant="outline">{chunkingInfo.strategy}</Badge>
                      <span className="text-muted-foreground">
                        {chunkingInfo.num_source_docs} 篇文档 → <strong className="text-foreground">{chunkingInfo.num_chunks}</strong> 个分块
                      </span>
                      <span className="text-muted-foreground">
                        (每块 ≤{chunkingInfo.chunk_size} 字)
                      </span>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Retrieval results */}
          <AnimatePresence>
            {retrievalResults.length > 0 && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <h3 className="text-xs font-medium text-muted-foreground mb-3">
                  📄 {isEnhanced && useChunking ? '检索到的文档片段' : '检索到的文档'}
                  {rerankData && ' (重排序前)'}
                </h3>
                <div className="space-y-2">
                  {retrievalResults.map((r, i) => {
                    const isChunk = 'text' in r && 'doc_title' in r;
                    return (
                      <Card key={i} className="overflow-hidden">
                        <CardContent className="py-3">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="text-xs font-mono text-muted-foreground">#{i + 1}</span>
                                <span className="text-sm font-medium truncate">
                                  {isChunk ? (r as EnhancedChunkResult).doc_title : (r as EmbeddingResult).title}
                                </span>
                                {isChunk && (
                                  <Badge variant="secondary" className="text-[10px]">
                                    {(r as EnhancedChunkResult).doc_id}
                                  </Badge>
                                )}
                              </div>
                              <p className="text-xs text-muted-foreground line-clamp-2">
                                {isChunk ? (r as EnhancedChunkResult).text : (r as EmbeddingResult).content}
                              </p>
                            </div>
                            <Badge variant="outline" className="shrink-0 text-xs font-mono">
                              {r.score.toFixed(4)}
                            </Badge>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Reranking comparison */}
          <AnimatePresence>
            {rerankData && isEnhanced && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <h3 className="text-xs font-medium text-purple-600 dark:text-purple-400 mb-3">🔄 重排序结果</h3>
                <div className="space-y-2">
                  {rerankData.after.map((item, i) => {
                    const change = item.rank_change;
                    return (
                      <Card key={i} className="overflow-hidden border-purple-200/50 dark:border-purple-500/20">
                        <CardContent className="py-3">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="text-xs font-mono text-muted-foreground">#{item.new_rank}</span>
                                <span className="text-sm font-medium truncate">{item.doc_title}</span>
                                <Badge variant="secondary" className="text-[10px]">{item.doc_id}</Badge>
                                {change > 0 && (
                                  <span className="text-xs text-green-600 dark:text-green-400 font-medium">↑{change}</span>
                                )}
                                {change < 0 && (
                                  <span className="text-xs text-red-500 font-medium">↓{Math.abs(change)}</span>
                                )}
                                {change === 0 && (
                                  <span className="text-xs text-muted-foreground">═</span>
                                )}
                              </div>
                              <p className="text-xs text-muted-foreground line-clamp-2">{item.text}</p>
                            </div>
                            <Badge variant="outline" className="shrink-0 text-xs font-mono text-purple-600 dark:text-purple-400">
                              CE: {item.score.toFixed(4)}
                            </Badge>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
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
                <h3 className="text-xs font-medium text-emerald-600 dark:text-emerald-400 mb-3">
                  🤖 {isEnhanced ? '增强 RAG 回答' : 'RAG 回答'}
                </h3>
                <div className="p-5 rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/30">
                  {isEnhanced ? (
                    <CitedText text={ragAnswer} sources={doneData?.sources ?? []} isStreaming={isStreaming} />
                  ) : (
                    <StreamingText text={ragAnswer} isStreaming={isStreaming} />
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Empty state */}
          {!loading && !ragAnswer && !error && retrievalResults.length === 0 && (
            <div className="text-center py-16 text-muted-foreground text-sm">
              {isEnhanced
                ? '调整 Pipeline 配置，输入问题体验增强 RAG 流程'
                : '输入问题开始体验基线 RAG 流程'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import SearchInput from '@/components/SearchInput';
import ScoreBar from '@/components/ScoreBar';
import ResultCard from '@/components/ResultCard';
import {
  embeddingSearch,
  hybridSearch,
  type EmbeddingResponse,
  type HybridResponse,
} from '@/lib/api';

type Tab = 'embedding' | 'hybrid';

export default function RouteBPage() {
  const [tab, setTab] = useState<Tab>('embedding');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [embeddingData, setEmbeddingData] = useState<EmbeddingResponse | null>(null);
  const [hybridData, setHybridData] = useState<HybridResponse | null>(null);

  const handleSearch = async (query: string) => {
    setLoading(true);
    setError(null);

    try {
      if (tab === 'embedding') {
        setEmbeddingData(null);
        const result = await embeddingSearch(query);
        setEmbeddingData(result);
      } else {
        setHybridData(null);
        const result = await hybridSearch(query);
        setHybridData(result);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '请求失败，请确认后端服务是否运行');
    } finally {
      setLoading(false);
    }
  };

  const embChartData = embeddingData?.results.map((r) => ({
    name: r.title.length > 12 ? r.title.slice(0, 12) + '…' : r.title,
    score: parseFloat(r.score.toFixed(4)),
  })) ?? [];

  const embMax = embeddingData?.results.reduce((m, r) => Math.max(m, r.score), 0) ?? 0;

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-xs font-mono px-2 py-0.5 rounded-md bg-purple-500/10 text-purple-400">
            路线 B
          </span>
          <h1 className="text-2xl font-bold">向量嵌入检索</h1>
        </div>
        <p className="text-white/40 text-sm leading-relaxed max-w-2xl">
          使用 Embedding 模型将文本映射到高维向量空间，通过余弦相似度进行语义匹配。
          混合搜索模式将 BM25 和向量检索结果融合，取长补短。
        </p>
      </motion.div>

      {/* Tab switcher */}
      <div className="flex gap-2 mb-6">
        {([
          { key: 'embedding' as Tab, label: '向量搜索' },
          { key: 'hybrid' as Tab, label: '混合搜索' },
        ]).map(({ key, label }) => (
          <button
            key={key}
            onClick={() => { setTab(key); setError(null); }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === key
                ? 'bg-purple-500/15 text-purple-300 border border-purple-500/30'
                : 'bg-white/5 text-white/40 border border-white/[0.06] hover:text-white/60'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Search */}
      <div className="mb-10">
        <SearchInput
          onSearch={handleSearch}
          loading={loading}
          placeholder={tab === 'embedding' ? '输入语义搜索内容...' : '输入混合搜索内容...'}
        />
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

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-24 rounded-xl bg-white/[0.03] animate-pulse" />
          ))}
        </div>
      )}

      {/* Embedding results */}
      <AnimatePresence>
        {tab === 'embedding' && embeddingData && !loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
            {/* Stats */}
            <div className="flex gap-4 mb-8 text-xs text-white/30">
              <span>耗时: <strong className="text-white/60">{embeddingData.elapsed_ms.toFixed(1)}ms</strong></span>
              <span>嵌入维度: <strong className="text-white/60">{embeddingData.embedding_dim}</strong></span>
              <span>结果数: <strong className="text-white/60">{embeddingData.results.length}</strong></span>
            </div>

            {/* Chart */}
            {embChartData.length > 0 && (
              <div className="mb-8 p-5 rounded-xl bg-white/[0.02] border border-white/[0.06]">
                <h3 className="text-xs font-medium text-white/40 mb-4">余弦相似度分布</h3>
                <ResponsiveContainer width="100%" height={embChartData.length * 50 + 20}>
                  <BarChart data={embChartData} layout="vertical" margin={{ left: 10, right: 30 }}>
                    <XAxis type="number" domain={[0, Math.min(embMax * 1.2, 1)]} tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="name" width={100} tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={{ background: '#1a1a1a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: '12px' }}
                      labelStyle={{ color: 'rgba(255,255,255,0.7)' }}
                      itemStyle={{ color: '#a855f7' }}
                    />
                    <Bar dataKey="score" radius={[0, 4, 4, 0]} animationDuration={800}>
                      {embChartData.map((_, index) => (
                        <Cell key={index} fill={`rgba(168, 85, 247, ${1 - index * 0.15})`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Result cards */}
            <div className="space-y-3">
              <h3 className="text-xs font-medium text-white/40 mb-2">检索结果</h3>
              {embeddingData.results.map((result, i) => (
                <ResultCard key={i} title={result.title} score={result.score} content={result.content} index={i} scoreLabel="相似度" />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hybrid results */}
      <AnimatePresence>
        {tab === 'hybrid' && hybridData && !loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
            {/* Stats */}
            <div className="flex gap-4 mb-8 text-xs text-white/30">
              <span>耗时: <strong className="text-white/60">{hybridData.elapsed_ms.toFixed(1)}ms</strong></span>
            </div>

            {/* Three-column comparison */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              {/* BM25 column */}
              <div className="space-y-3">
                <h3 className="text-xs font-medium text-blue-400 mb-3 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-400" />
                  BM25 结果
                </h3>
                {hybridData.bm25_results.map((r, i) => {
                  const maxS = hybridData.bm25_results[0]?.score || 1;
                  return <ScoreBar key={i} label={r.title} score={r.score} maxScore={maxS} index={i} color="#3b82f6" />;
                })}
              </div>

              {/* Embedding column */}
              <div className="space-y-3">
                <h3 className="text-xs font-medium text-purple-400 mb-3 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-purple-400" />
                  向量结果
                </h3>
                {hybridData.embedding_results.map((r, i) => {
                  const maxS = hybridData.embedding_results[0]?.score || 1;
                  return <ScoreBar key={i} label={r.title} score={r.score} maxScore={maxS} index={i} color="#a855f7" />;
                })}
              </div>

              {/* Hybrid column */}
              <div className="space-y-3">
                <h3 className="text-xs font-medium text-emerald-400 mb-3 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  混合结果
                </h3>
                {hybridData.hybrid_results.map((r, i) => {
                  const maxS = hybridData.hybrid_results[0]?.score || 1;
                  return <ScoreBar key={i} label={r.title} score={r.score} maxScore={maxS} index={i} color="#10b981" />;
                })}
              </div>
            </div>

            {/* Hybrid result cards */}
            <div className="space-y-3">
              <h3 className="text-xs font-medium text-white/40 mb-2">混合检索详细结果</h3>
              {hybridData.hybrid_results.map((result, i) => (
                <ResultCard key={i} title={result.title} score={result.score} content={result.content} index={i} scoreLabel="混合分" />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

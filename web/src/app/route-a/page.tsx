'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import SearchInput from '@/components/SearchInput';
import ResultCard from '@/components/ResultCard';
import { bm25Search, type BM25Response } from '@/lib/api';

export default function RouteAPage() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<BM25Response | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [animatedTokens, setAnimatedTokens] = useState<string[]>([]);

  const handleSearch = async (query: string) => {
    setLoading(true);
    setError(null);
    setData(null);
    setAnimatedTokens([]);

    try {
      const result = await bm25Search(query);
      setData(result);

      // Animate tokens appearing one by one
      result.query_tokens.forEach((token, i) => {
        setTimeout(() => {
          setAnimatedTokens((prev) => [...prev, token]);
        }, i * 150);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : '搜索请求失败，请确认后端服务是否运行');
    } finally {
      setLoading(false);
    }
  };

  const chartData = data?.results.map((r) => ({
    name: r.title.length > 12 ? r.title.slice(0, 12) + '…' : r.title,
    score: parseFloat(r.score.toFixed(4)),
  })) ?? [];

  const maxScore = data?.results.reduce((max, r) => Math.max(max, r.score), 0) ?? 0;

  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-10"
      >
        <div className="flex items-center gap-3 mb-3">
          <span className="text-xs font-mono px-2 py-0.5 rounded-md bg-blue-50 text-blue-600">
            路线 A
          </span>
          <h1 className="text-2xl font-bold">BM25 关键词搜索</h1>
        </div>
        <p className="text-slate-500 text-sm leading-relaxed max-w-2xl">
          BM25 是经典的关键词检索算法，基于词频（TF）和逆文档频率（IDF）对文档评分。
          输入查询后，你可以看到分词过程和每个文档的匹配得分。
        </p>
      </motion.div>

      {/* Search */}
      <div className="mb-10">
        <SearchInput onSearch={handleSearch} loading={loading} placeholder="输入关键词搜索..." />
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mb-8 p-4 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm"
          >
            ⚠️ {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-24 rounded-xl bg-slate-100 animate-pulse" />
          ))}
        </div>
      )}

      {/* Results */}
      <AnimatePresence>
        {data && !loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
          >
            {/* Stats */}
            <div className="flex gap-4 mb-8 text-xs text-slate-400">
              <span>耗时: <strong className="text-slate-600">{data.elapsed_ms.toFixed(1)}ms</strong></span>
              <span>结果数: <strong className="text-slate-600">{data.results.length}</strong></span>
            </div>

            {/* Token animation */}
            <div className="mb-8">
              <h3 className="text-xs font-medium text-slate-500 mb-3">查询分词结果</h3>
              <div className="flex flex-wrap gap-2">
                <AnimatePresence>
                  {animatedTokens.map((token, i) => (
                    <motion.span
                      key={`${token}-${i}`}
                      initial={{ opacity: 0, scale: 0.7, y: 10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      className="px-3 py-1.5 rounded-lg bg-blue-50 text-blue-600 text-sm font-mono border border-blue-200"
                    >
                      {token}
                    </motion.span>
                  ))}
                </AnimatePresence>
              </div>
            </div>

            {/* Score chart */}
            {chartData.length > 0 && (
              <div className="mb-8 p-5 rounded-xl bg-white border border-slate-200 shadow-sm">
                <h3 className="text-xs font-medium text-slate-500 mb-4">BM25 得分分布</h3>
                <ResponsiveContainer width="100%" height={chartData.length * 50 + 20}>
                  <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
                    <XAxis type="number" domain={[0, maxScore * 1.1]} tick={{ fill: 'rgba(100,116,139,0.7)', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="name" width={100} tick={{ fill: 'rgba(71,85,105,0.8)', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px', fontSize: '12px' }}
                      labelStyle={{ color: '#334155' }}
                      itemStyle={{ color: '#3b82f6' }}
                    />
                    <Bar dataKey="score" radius={[0, 4, 4, 0]} animationDuration={800}>
                      {chartData.map((_, index) => (
                        <Cell key={index} fill={`rgba(59, 130, 246, ${1 - index * 0.15})`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Result cards */}
            <div className="space-y-3">
              <h3 className="text-xs font-medium text-slate-500 mb-2">检索结果</h3>
              {data.results.map((result, i) => (
                <ResultCard
                  key={i}
                  title={result.title}
                  score={result.score}
                  content={result.content}
                  index={i}
                  scoreLabel="BM25"
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

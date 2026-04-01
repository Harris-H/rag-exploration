'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HelpCircle, X } from 'lucide-react';
import SearchInput from '@/components/SearchInput';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { chunkingCompare, type ChunkingCompareResponse } from '@/lib/api';

export default function ChunkingTab() {
  const [loading, setLoading] = useState(false);
  const [showExplain, setShowExplain] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ChunkingCompareResponse | null>(null);
  const [chunkSize, setChunkSize] = useState(150);

  const handleSearch = async (query: string) => {
    setLoading(true);
    setError(null);
    try {
      setData(null);
      const result = await chunkingCompare(query, chunkSize, 50);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : '请求失败');
    } finally {
      setLoading(false);
    }
  };

  const strategyLabels: Record<string, { name: string; color: string; emoji: string }> = {
    fixed: { name: '固定长度', color: '#3b82f6', emoji: '📏' },
    sentence: { name: '句子边界', color: '#a855f7', emoji: '📝' },
    sliding: { name: '滑动窗口', color: '#10b981', emoji: '🔄' },
    recursive: { name: '递归分块', color: '#f59e0b', emoji: '🌲' },
  };

  const chartData = data ? Object.entries(data.strategies).map(([key, s]) => ({
    name: strategyLabels[key]?.name ?? key,
    chunks: s.num_chunks,
    avg_len: Math.round(s.avg_chunk_len),
    color: strategyLabels[key]?.color ?? '#666',
  })) : [];

  return (
    <div>
      {/* Controls */}
      <div className="my-6 space-y-4">
        <div className="flex items-center gap-3">
          <SearchInput onSearch={handleSearch} loading={loading} placeholder="输入查询，对比不同分块策略的检索效果..." />
          <button
            onClick={() => setShowExplain(!showExplain)}
            className="shrink-0 p-2 rounded-full text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            title="了解分块策略的区别"
          >
            <HelpCircle className="size-5" />
          </button>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <label className="text-muted-foreground">分块大小:</label>
          <input type="range" min={50} max={500} step={50} value={chunkSize}
            onChange={(e) => setChunkSize(Number(e.target.value))}
            className="w-48 accent-purple-600" />
          <Badge variant="outline" className="font-mono">{chunkSize} 字符</Badge>
        </div>
      </div>

      {/* Explanation panel */}
      <AnimatePresence>
        {showExplain && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-6 overflow-hidden"
          >
            <Card className="bg-purple-50/50 dark:bg-purple-500/5 border-purple-200 dark:border-purple-500/20">
              <CardContent className="py-4 text-sm leading-relaxed">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-medium text-purple-700 dark:text-purple-400">💡 为什么分块策略会影响检索效果？</h4>
                  <button onClick={() => setShowExplain(false)} className="text-muted-foreground hover:text-foreground">
                    <X className="size-4" />
                  </button>
                </div>
                <div className="space-y-2 text-muted-foreground">
                  <p>在 RAG 系统中，长文档需要先切分成小块再向量化。不同的分块方式会直接影响检索结果：</p>
                  <p><strong className="text-foreground">📏 固定长度</strong>：按字符数硬切，简单快速，但可能把一句话从中间截断，导致语义不完整。</p>
                  <p><strong className="text-foreground">📝 句子边界</strong>：在句号、问号等标点处切分，保证每块都是完整的句子，语义更连贯。</p>
                  <p><strong className="text-foreground">🔄 滑动窗口</strong>：相邻块之间有重叠区域（Overlap），减少边界处的信息丢失，但会产生更多的块。</p>
                  <p><strong className="text-foreground">🌲 递归分块</strong>：先按段落分，太长再按句子分，太长再按固定长度分。自适应能力最强，是 LangChain 等框架的默认策略。</p>
                  <p className="text-xs pt-1 border-t border-purple-200/50 dark:border-purple-500/10">💡 拖动上方滑块调整分块大小，观察不同大小对各策略的分块数量和检索效果的影响。</p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="mb-8 p-4 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-sm">
            ⚠️ {error}
          </motion.div>
        )}
      </AnimatePresence>

      {loading && (
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => <div key={i} className="h-24 rounded-xl bg-muted animate-pulse" />)}
        </div>
      )}

      <AnimatePresence>
        {data && !loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
            {/* Stats */}
            <div className="flex gap-4 mb-6 text-xs text-muted-foreground">
              <span>耗时: <strong className="text-foreground">{data.elapsed_ms.toFixed(1)}ms</strong></span>
              <span>原文长度: <strong className="text-foreground">{data.source_text_length.toLocaleString()} 字符</strong></span>
            </div>

            {/* Overview chart: chunks count per strategy */}
            {chartData.length > 0 && (
              <Card className="mb-8">
                <CardContent>
                  <h3 className="text-xs font-medium text-muted-foreground mb-4">各策略分块数量 & 平均长度</h3>
                  <ResponsiveContainer width="100%" height={chartData.length * 60 + 20}>
                    <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
                      <XAxis type="number" tick={{ fill: 'var(--color-muted-foreground)', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis type="category" dataKey="name" width={80} tick={{ fill: 'var(--color-foreground)', fontSize: 12 }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-border)', borderRadius: '8px', fontSize: '12px' }} />
                      <Bar dataKey="chunks" name="分块数" radius={[0, 4, 4, 0]} animationDuration={800}>
                        {chartData.map((d, i) => <Cell key={i} fill={d.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}

            {/* Strategy detail cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(data.strategies).map(([key, strategy]) => {
                const meta = strategyLabels[key] ?? { name: key, color: '#666', emoji: '📄' };
                return (
                  <Card key={key}>
                    <CardContent>
                      <div className="flex items-center gap-2 mb-3">
                        <span>{meta.emoji}</span>
                        <h3 className="text-sm font-medium">{meta.name}</h3>
                        <Badge variant="outline" className="ml-auto font-mono text-xs">{strategy.num_chunks} 块</Badge>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-xs text-muted-foreground mb-4">
                        <div>平均: <strong className="text-foreground">{strategy.avg_chunk_len.toFixed(0)}</strong></div>
                        <div>最小: <strong className="text-foreground">{strategy.min_chunk_len}</strong></div>
                        <div>最大: <strong className="text-foreground">{strategy.max_chunk_len}</strong></div>
                      </div>
                      <h4 className="text-xs text-muted-foreground mb-2">Top-3 匹配块</h4>
                      <div className="space-y-2">
                        {strategy.top_chunks.map((chunk, i) => (
                          <div key={i} className="group p-2.5 rounded-lg bg-muted/50 text-xs cursor-default">
                            <div className="flex justify-between mb-1">
                              <span className="font-mono" style={{ color: meta.color }}>
                                相似度: {chunk.score.toFixed(4)}
                              </span>
                              <span className="text-muted-foreground">{chunk.full_length}字</span>
                            </div>
                            <p className="text-muted-foreground line-clamp-2 group-hover:line-clamp-none leading-relaxed transition-all duration-300">{chunk.text}</p>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

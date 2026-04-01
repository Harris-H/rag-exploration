'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Loader2, Play, HelpCircle, X } from 'lucide-react';
import { evaluateAll, type EvalAllResponse } from '@/lib/api';

export default function EvaluationTab() {
  const [loading, setLoading] = useState(false);
  const [showExplain, setShowExplain] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<EvalAllResponse | null>(null);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(null);
      const result = await evaluateAll(3);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : '请求失败');
    } finally {
      setLoading(false);
    }
  };

  const methodColors: Record<string, string> = {
    BM25: '#3b82f6',
    Embedding: '#a855f7',
    Hybrid: '#10b981',
  };
  const methodLabels: Record<string, string> = {
    BM25: 'BM25',
    Embedding: 'Embedding',
    Hybrid: 'Hybrid',
  };

  const metricNames = ['mrr', 'precision_at_k', 'recall_at_k', 'ndcg_at_k'] as const;
  const metricLabels: Record<string, string> = {
    mrr: 'MRR',
    precision_at_k: 'P@3',
    recall_at_k: 'R@3',
    ndcg_at_k: 'NDCG@3',
  };

  const chartData = data ? metricNames.map(m => ({
    metric: metricLabels[m],
    BM25: parseFloat((data.aggregate.BM25?.[m] ?? 0).toFixed(3)),
    Embedding: parseFloat((data.aggregate.Embedding?.[m] ?? 0).toFixed(3)),
    Hybrid: parseFloat((data.aggregate.Hybrid?.[m] ?? 0).toFixed(3)),
  })) : [];

  return (
    <div>
      {/* Run button */}
      <div className="my-6">
        <div className="flex items-center gap-3">
          <Button onClick={handleRun} disabled={loading}
            className="h-11 px-6 rounded-xl bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white gap-2">
            {loading ? <Loader2 className="size-5 animate-spin" /> : <Play className="size-5" />}
            运行评估（8 条测试查询 × 3 种方法）
          </Button>
          <button
            onClick={() => setShowExplain(!showExplain)}
            className="shrink-0 p-2 rounded-full text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            title="了解评估指标含义"
          >
            <HelpCircle className="size-5" />
          </button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">使用预设的 8 条测试查询和人工标注的相关文档，评估 BM25、Embedding、Hybrid 三种检索方法。</p>
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
            <Card className="bg-emerald-50/50 dark:bg-emerald-500/5 border-emerald-200 dark:border-emerald-500/20">
              <CardContent className="py-4 text-sm leading-relaxed">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-medium text-emerald-700 dark:text-emerald-400">💡 这些评估指标是什么意思？</h4>
                  <button onClick={() => setShowExplain(false)} className="text-muted-foreground hover:text-foreground">
                    <X className="size-4" />
                  </button>
                </div>
                <div className="space-y-2 text-muted-foreground">
                  <p><strong className="text-foreground">MRR（Mean Reciprocal Rank）</strong>：第一个正确结果出现在第几位？MRR=1 表示每次搜索第一条就是对的；MRR=0.5 表示平均在第 2 位才找到。</p>
                  <p><strong className="text-foreground">P@K（Precision at K）</strong>：前 K 条结果中有多少比例是相关的？P@3=0.67 表示返回的 3 条结果中有 2 条是正确的。</p>
                  <p><strong className="text-foreground">R@K（Recall at K）</strong>：所有相关文档中，有多少被包含在前 K 条结果里？R@3=1.0 表示所有相关文档都被找到了。</p>
                  <p><strong className="text-foreground">NDCG@K（Normalized DCG）</strong>：不仅看找没找到，还看排得对不对。把正确结果排在第 1 位比排在第 3 位得分更高。NDCG=1.0 是完美排序。</p>
                  <p className="text-xs pt-1 border-t border-emerald-200/50 dark:border-emerald-500/10">🏆 表格中带有 🏆 标记的是该指标下表现最好的方法。一般来说，Hybrid（混合检索）在大多数指标上表现最均衡。</p>
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
              <span>测试查询: <strong className="text-foreground">{data.num_queries}</strong></span>
              <span>K = <strong className="text-foreground">{data.k}</strong></span>
            </div>

            {/* Aggregate chart */}
            {chartData.length > 0 && (
              <Card className="mb-8">
                <CardContent>
                  <h3 className="text-xs font-medium text-muted-foreground mb-4">综合评估指标对比</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={chartData} margin={{ left: 10, right: 10 }}>
                      <XAxis dataKey="metric" tick={{ fill: 'var(--color-foreground)', fontSize: 12 }} axisLine={false} tickLine={false} />
                      <YAxis domain={[0, 1]} tick={{ fill: 'var(--color-muted-foreground)', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-border)', borderRadius: '8px', fontSize: '12px' }} />
                      <Legend />
                      <Bar dataKey="BM25" name="BM25" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Embedding" name="Embedding" fill="#a855f7" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Hybrid" name="Hybrid" fill="#10b981" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}

            {/* Aggregate summary table */}
            <Card className="mb-8">
              <CardContent>
                <h3 className="text-xs font-medium text-muted-foreground mb-4">综合指标汇总</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-2 px-3 text-muted-foreground font-medium">指标</th>
                        {Object.keys(methodLabels).map(m => (
                          <th key={m} className="text-right py-2 px-3 font-medium" style={{ color: methodColors[m] }}>
                            {methodLabels[m]}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {metricNames.map(metric => {
                        const values = Object.keys(methodLabels).map(m => data.aggregate[m]?.[metric] ?? 0);
                        const maxVal = Math.max(...values);
                        return (
                          <tr key={metric} className="border-b border-border/50">
                            <td className="py-2 px-3 text-muted-foreground">{metricLabels[metric]}</td>
                            {Object.keys(methodLabels).map((m, i) => {
                              const val = values[i];
                              const isBest = val === maxVal && maxVal > 0;
                              return (
                                <td key={m} className={`text-right py-2 px-3 font-mono ${isBest ? 'font-bold text-foreground' : 'text-muted-foreground'}`}>
                                  {val.toFixed(3)}{isBest ? ' 🏆' : ''}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Per-query details */}
            <h3 className="text-xs font-medium text-muted-foreground mb-3">逐查询详情</h3>
            <div className="space-y-3">
              {data.per_query.map((q, qi) => (
                <Card key={qi}>
                  <CardContent className="py-3">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div>
                        <p className="text-sm font-medium">{q.query}</p>
                        <p className="text-xs text-muted-foreground">{q.description} · 相关文档: {q.relevant_docs.join(', ')}</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-3 text-xs">
                      {Object.entries(q.methods).map(([method, info]) => {
                        const best = Math.max(...Object.values(q.methods).map(m => m.metrics.ndcg_at_k));
                        const isBest = info.metrics.ndcg_at_k === best;
                        return (
                          <div key={method} className={`p-2 rounded-lg ${isBest ? 'bg-emerald-50 dark:bg-emerald-500/10 ring-1 ring-emerald-200 dark:ring-emerald-500/30' : 'bg-muted/50'}`}>
                            <span className="font-medium" style={{ color: methodColors[method] }}>{methodLabels[method]}</span>
                            <div className="mt-1 space-y-0.5 text-muted-foreground font-mono">
                              <div>MRR: {info.metrics.mrr.toFixed(2)}</div>
                              <div>NDCG: {info.metrics.ndcg_at_k.toFixed(2)}</div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HelpCircle, X } from 'lucide-react';
import SearchInput from '@/components/SearchInput';
import ResultCard from '@/components/ResultCard';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { rerankSearch, type RerankResponse } from '@/lib/api';

export default function RerankingTab() {
  const [loading, setLoading] = useState(false);
  const [showExplain, setShowExplain] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<RerankResponse | null>(null);

  const handleSearch = async (query: string) => {
    setLoading(true);
    setError(null);
    try {
      setData(null);
      const result = await rerankSearch(query, 5);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : '请求失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="my-6 flex items-center gap-3">
        <SearchInput onSearch={handleSearch} loading={loading} placeholder="输入查询，对比重排序前后的结果..." />
        <button
          onClick={() => setShowExplain(!showExplain)}
          className="shrink-0 p-2 rounded-full text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          title="了解为什么重排序更好"
        >
          <HelpCircle className="size-5" />
        </button>
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
            <Card className="bg-blue-50/50 dark:bg-blue-500/5 border-blue-200 dark:border-blue-500/20">
              <CardContent className="py-4 text-sm leading-relaxed">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-medium text-blue-700 dark:text-blue-400">💡 为什么 Cross-Encoder 重排序更准确？</h4>
                  <button onClick={() => setShowExplain(false)} className="text-muted-foreground hover:text-foreground">
                    <X className="size-4" />
                  </button>
                </div>
                <div className="space-y-2 text-muted-foreground">
                  <p><strong className="text-foreground">BM25（关键词匹配）</strong>：只统计词频。搜"重排序"时，只要文档含"排序"二字就给高分，容易被无关文档骗。</p>
                  <p><strong className="text-foreground">Bi-Encoder（向量检索）</strong>：将查询和文档分别编码为向量再算相似度。能理解同义词，但查询和文档之间没有直接交互。</p>
                  <p><strong className="text-foreground">Cross-Encoder（交叉编码器）</strong>：将查询和文档<strong className="text-blue-600 dark:text-blue-400">拼接在一起</strong>输入 Transformer，让每个词都能互相 Attention。它能理解"重排序"和"Reranking"是同一概念，也能看穿 Google 搜索引擎虽然提到"排序"但讨论的是网页排名，不是检索重排。</p>
                  <p className="text-xs pt-1 border-t border-blue-200/50 dark:border-blue-500/10">⚡ 代价：Cross-Encoder 需要对每个候选文档做一次完整前向传播，计算量大，因此只用于对 Top-K 初检结果进行二次重排。</p>
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

      {/* Loading */}
      {loading && (
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => <div key={i} className="h-24 rounded-xl bg-muted animate-pulse" />)}
        </div>
      )}

      {/* Results */}
      <AnimatePresence>
        {data && !loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
            {/* Stats */}
            <div className="flex gap-4 mb-6 text-xs text-muted-foreground">
              <span>耗时: <strong className="text-foreground">{data.elapsed_ms.toFixed(1)}ms</strong></span>
              <span>重排模型: <strong className="text-foreground">{data.reranker_model}</strong></span>
            </div>

            {/* Before vs After side-by-side */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              {/* Before reranking */}
              <div>
                <h3 className="text-xs font-medium text-orange-600 dark:text-orange-400 mb-3 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-orange-500" />
                  重排序前 (RRF 混合检索)
                </h3>
                <div className="space-y-2">
                  {data.before_rerank.map((r, i) => (
                    <Card key={i} className="hover:shadow-sm transition-all">
                      <CardContent className="py-3 px-4">
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2 min-w-0">
                            <Badge variant="outline" className="shrink-0 font-mono text-xs">#{r.rank}</Badge>
                            <span className="text-sm font-medium truncate">{r.title}</span>
                          </div>
                          <span className="text-xs font-mono text-muted-foreground shrink-0">{r.score.toFixed(4)}</span>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>

              {/* After reranking */}
              <div>
                <h3 className="text-xs font-medium text-emerald-600 dark:text-emerald-400 mb-3 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  重排序后 (Cross-Encoder)
                </h3>
                <div className="space-y-2">
                  {data.after_rerank.map((r, i) => {
                    const change = r.rank_change;
                    const changeLabel = change > 0 ? `↑${change}` : change < 0 ? `↓${Math.abs(change)}` : '═';
                    const changeColor = change > 0 ? 'text-emerald-600 dark:text-emerald-400' : change < 0 ? 'text-red-500' : 'text-muted-foreground';
                    return (
                      <Card key={i} className="hover:shadow-sm transition-all">
                        <CardContent className="py-3 px-4">
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2 min-w-0">
                              <Badge variant="outline" className="shrink-0 font-mono text-xs">#{r.new_rank}</Badge>
                              <span className="text-sm font-medium truncate">{r.title}</span>
                              <span className={`text-xs font-bold ${changeColor}`}>{changeLabel}</span>
                            </div>
                            <span className="text-xs font-mono text-muted-foreground shrink-0">{r.score.toFixed(4)}</span>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Detailed results */}
            <div className="space-y-3">
              <h3 className="text-xs font-medium text-muted-foreground mb-2">重排序后详细结果</h3>
              {data.after_rerank.slice(0, 3).map((r, i) => (
                <ResultCard key={i} title={r.title} score={r.score} content={r.content} index={i} scoreLabel="CE分数" />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

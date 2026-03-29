'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';

const routes = [
  {
    href: '/route-a',
    label: '路线 A',
    title: 'BM25 关键词搜索',
    description: '经典的 TF-IDF 变体算法，通过词频和逆文档频率进行文本匹配。理解信息检索的基础。',
    icon: '🔤',
    color: 'from-blue-50 to-white',
    border: 'hover:border-blue-300',
    badge: 'bg-blue-50 text-blue-600',
  },
  {
    href: '/route-b',
    label: '路线 B',
    title: '向量嵌入检索',
    description: '使用语义嵌入模型将文本映射到高维空间，通过余弦相似度发现语义关联。支持混合搜索对比。',
    icon: '🧮',
    color: 'from-purple-50 to-white',
    border: 'hover:border-purple-300',
    badge: 'bg-purple-50 text-purple-600',
  },
  {
    href: '/route-c',
    label: '路线 C',
    title: '完整 RAG 管道',
    description: '将检索结果注入 Prompt，由本地 LLM 生成答案。体验完整的检索增强生成流程，支持流式输出。',
    icon: '🤖',
    color: 'from-emerald-50 to-white',
    border: 'hover:border-emerald-300',
    badge: 'bg-emerald-50 text-emerald-600',
  },
];

const containerVariants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.15 } },
};

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' as const } },
};

export default function Home() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-20">
      {/* Hero */}
      <motion.div
        className="text-center mb-20"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight mb-4">
          RAG <span className="text-blue-600">探索之旅</span>
        </h1>
        <p className="text-slate-500 text-lg max-w-xl mx-auto leading-relaxed">
          从关键词搜索到向量检索，再到完整的检索增强生成管道。
          <br />
          渐进式理解 RAG 的每一个核心环节。
        </p>
      </motion.div>

      {/* What is RAG */}
      <motion.div
        className="mb-16 p-6 rounded-2xl bg-white border border-slate-200 shadow-sm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.6 }}
      >
        <h2 className="text-sm font-medium text-slate-600 mb-3 flex items-center gap-2">
          <span className="w-1 h-1 rounded-full bg-blue-600" />
          什么是 RAG？
        </h2>
        <p className="text-slate-500 text-sm leading-relaxed">
          <strong className="text-slate-700">检索增强生成（Retrieval-Augmented Generation）</strong>
          是一种结合信息检索与大语言模型的技术。它先从知识库中检索相关文档，
          然后将检索结果作为上下文注入 Prompt，让 LLM 基于实际数据生成更准确的回答。
          这种方式有效减少了幻觉问题，并让模型能够访问最新的领域知识。
        </p>
      </motion.div>

      {/* Route cards */}
      <motion.div
        className="grid gap-4"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        {routes.map((route) => (
          <motion.div key={route.href} variants={itemVariants}>
            <Link href={route.href} className="block group">
              <div
                className={`relative p-6 rounded-2xl border border-slate-200 ${route.border}
                            bg-gradient-to-br ${route.color} transition-all duration-300
                            hover:translate-x-1 shadow-sm hover:shadow-md`}
              >
                <div className="flex items-start gap-4">
                  <span className="text-2xl">{route.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-2">
                      <span className={`text-xs font-mono px-2 py-0.5 rounded-md ${route.badge}`}>
                        {route.label}
                      </span>
                      <h3 className="text-slate-900 font-semibold">{route.title}</h3>
                    </div>
                    <p className="text-slate-500 text-sm leading-relaxed">
                      {route.description}
                    </p>
                  </div>
                  <svg
                    className="w-5 h-5 text-slate-300 group-hover:text-slate-500 group-hover:translate-x-1 transition-all shrink-0 mt-1"
                    fill="none" stroke="currentColor" viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </div>
            </Link>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}

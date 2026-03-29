'use client';

import { motion } from 'framer-motion';

interface ResultCardProps {
  title: string;
  score: number;
  content: string;
  index?: number;
  scoreLabel?: string;
}

export default function ResultCard({
  title,
  score,
  content,
  index = 0,
  scoreLabel = '得分',
}: ResultCardProps) {
  return (
    <motion.div
      className="p-4 rounded-xl bg-white border border-slate-200 hover:border-slate-300 shadow-sm hover:shadow-md transition-all"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.08 }}
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <h3 className="text-slate-900 font-medium text-sm leading-snug">{title}</h3>
        <span className="shrink-0 px-2 py-0.5 rounded-md bg-amber-50 text-amber-600 text-xs font-mono">
          {scoreLabel}: {score.toFixed(4)}
        </span>
      </div>
      <p className="text-slate-600 text-sm leading-relaxed line-clamp-3">{content}</p>
    </motion.div>
  );
}

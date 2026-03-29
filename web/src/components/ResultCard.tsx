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
      className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.06] hover:border-white/10 transition-colors"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.08 }}
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <h3 className="text-white font-medium text-sm leading-snug">{title}</h3>
        <span className="shrink-0 px-2 py-0.5 rounded-md bg-amber-500/15 text-amber-400 text-xs font-mono">
          {scoreLabel}: {score.toFixed(4)}
        </span>
      </div>
      <p className="text-white/50 text-sm leading-relaxed line-clamp-3">{content}</p>
    </motion.div>
  );
}

'use client';

import { motion } from 'framer-motion';

interface ScoreBarProps {
  label: string;
  score: number;
  maxScore: number;
  index?: number;
  color?: string;
}

export default function ScoreBar({
  label,
  score,
  maxScore,
  index = 0,
  color = '#3b82f6',
}: ScoreBarProps) {
  const percentage = maxScore > 0 ? (score / maxScore) * 100 : 0;

  return (
    <div className="flex items-center gap-3 group">
      <span className="text-sm text-slate-600 w-32 truncate shrink-0" title={label}>
        {label}
      </span>
      <div className="flex-1 h-7 bg-slate-100 rounded-lg overflow-hidden relative">
        <motion.div
          className="h-full rounded-lg"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(percentage, 2)}%` }}
          transition={{ duration: 0.6, delay: index * 0.1, ease: 'easeOut' }}
        />
        <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-slate-700 font-mono">
          {score.toFixed(4)}
        </span>
      </div>
    </div>
  );
}

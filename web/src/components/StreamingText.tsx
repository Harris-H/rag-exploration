'use client';

import { motion } from 'framer-motion';

interface StreamingTextProps {
  text: string;
  isStreaming?: boolean;
  className?: string;
}

export default function StreamingText({
  text,
  isStreaming = false,
  className = '',
}: StreamingTextProps) {
  return (
    <div className={`relative ${className}`}>
      <p className="text-white/80 leading-relaxed whitespace-pre-wrap break-words">
        {text}
        {isStreaming && (
          <motion.span
            className="inline-block w-2 h-5 bg-blue-400 ml-0.5 align-middle rounded-sm"
            animate={{ opacity: [1, 0] }}
            transition={{ duration: 0.6, repeat: Infinity, repeatType: 'reverse' }}
          />
        )}
      </p>
      {!text && isStreaming && (
        <div className="flex items-center gap-2 text-white/30 text-sm">
          <motion.span
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            等待 LLM 生成中...
          </motion.span>
        </div>
      )}
    </div>
  );
}

'use client';

import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import type { CitationSource } from '@/lib/api';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface CitedTextProps {
  text: string;
  sources: CitationSource[];
  isStreaming?: boolean;
  className?: string;
}

/**
 * Renders text with inline citation markers [1], [2], etc.
 * as styled superscript badges with tooltips showing the source document.
 */
export default function CitedText({
  text,
  sources,
  isStreaming = false,
  className = '',
}: CitedTextProps) {
  const sourceMap = useMemo(() => {
    const map = new Map<number, CitationSource>();
    for (const s of sources) {
      map.set(s.index, s);
    }
    return map;
  }, [sources]);

  // Parse text and split by citation markers [N]
  const segments = useMemo(() => {
    if (!text) return [];
    const parts: { type: 'text' | 'cite'; content: string; index?: number }[] = [];
    // Match [1], [2], [3] etc. - also handle [1][2] consecutive
    const regex = /\[(\d+)\]/g;
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(text)) !== null) {
      // Text before the citation
      if (match.index > lastIndex) {
        parts.push({ type: 'text', content: text.slice(lastIndex, match.index) });
      }
      parts.push({ type: 'cite', content: match[0], index: parseInt(match[1], 10) });
      lastIndex = regex.lastIndex;
    }
    // Remaining text
    if (lastIndex < text.length) {
      parts.push({ type: 'text', content: text.slice(lastIndex) });
    }
    return parts;
  }, [text]);

  // Collect cited source indices for the reference list
  const citedIndices = useMemo(() => {
    const indices = new Set<number>();
    for (const seg of segments) {
      if (seg.type === 'cite' && seg.index && sourceMap.has(seg.index)) {
        indices.add(seg.index);
      }
    }
    return Array.from(indices).sort((a, b) => a - b);
  }, [segments, sourceMap]);

  return (
    <div className={`relative ${className}`}>
      <TooltipProvider delayDuration={200}>
        <p className="text-foreground leading-relaxed whitespace-pre-wrap break-words">
          {segments.map((seg, i) => {
            if (seg.type === 'text') {
              return <span key={i}>{seg.content}</span>;
            }
            const source = seg.index ? sourceMap.get(seg.index) : null;
            if (!source) {
              // Unknown citation number, render as plain text
              return <span key={i}>{seg.content}</span>;
            }
            return (
              <Tooltip key={i}>
                <TooltipTrigger asChild>
                  <sup className="inline-flex items-center justify-center w-4 h-4 mx-0.5 text-[10px] font-bold rounded-full bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 cursor-help hover:bg-emerald-200 dark:hover:bg-emerald-500/30 transition-colors align-super leading-none">
                    {seg.index}
                  </sup>
                </TooltipTrigger>
                <TooltipContent side="top" className="max-w-xs">
                  <p className="text-xs font-medium">{source.doc_title}</p>
                  <p className="text-[10px] text-muted-foreground">{source.doc_id}</p>
                </TooltipContent>
              </Tooltip>
            );
          })}
          {isStreaming && (
            <motion.span
              className="inline-block w-2 h-5 bg-primary ml-0.5 align-middle rounded-sm"
              animate={{ opacity: [1, 0] }}
              transition={{ duration: 0.6, repeat: Infinity, repeatType: 'reverse' }}
            />
          )}
        </p>
      </TooltipProvider>

      {!text && isStreaming && (
        <div className="flex items-center gap-2 text-muted-foreground text-sm">
          <motion.span
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            等待 LLM 生成中...
          </motion.span>
        </div>
      )}

      {/* Reference list */}
      {citedIndices.length > 0 && !isStreaming && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-4 pt-3 border-t border-emerald-200/50 dark:border-emerald-500/20"
        >
          <p className="text-[10px] font-medium text-muted-foreground mb-1.5">参考来源</p>
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {citedIndices.map((idx) => {
              const source = sourceMap.get(idx);
              if (!source) return null;
              return (
                <span key={idx} className="text-xs text-muted-foreground">
                  <sup className="inline-flex items-center justify-center w-3.5 h-3.5 mr-0.5 text-[9px] font-bold rounded-full bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 align-super leading-none">
                    {idx}
                  </sup>
                  {source.doc_title}
                </span>
              );
            })}
          </div>
        </motion.div>
      )}
    </div>
  );
}

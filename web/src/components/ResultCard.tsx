'use client';

import { motion } from 'framer-motion';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.08 }}
    >
      <Card className="hover:shadow-md transition-all">
        <CardContent>
          <div className="flex items-start justify-between gap-3 mb-2">
            <h3 className="text-foreground font-medium text-sm leading-snug">{title}</h3>
            <Badge variant="outline" className="shrink-0 font-mono text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-500/30 bg-amber-50 dark:bg-amber-500/10">
              {scoreLabel}: {score.toFixed(4)}
            </Badge>
          </div>
          <p className="text-muted-foreground text-sm leading-relaxed line-clamp-3">{content}</p>
        </CardContent>
      </Card>
    </motion.div>
  );
}

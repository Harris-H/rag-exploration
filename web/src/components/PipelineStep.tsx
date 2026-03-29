'use client';

import { motion } from 'framer-motion';

type StepStatus = 'idle' | 'active' | 'done';

interface PipelineStepProps {
  number: number;
  title: string;
  description: string;
  status: StepStatus;
  isLast?: boolean;
}

const statusStyles: Record<StepStatus, { ring: string; bg: string; text: string }> = {
  idle: { ring: 'border-slate-200', bg: 'bg-slate-50', text: 'text-slate-400' },
  active: { ring: 'border-blue-500', bg: 'bg-blue-50', text: 'text-blue-600' },
  done: { ring: 'border-green-500', bg: 'bg-green-50', text: 'text-green-600' },
};

export default function PipelineStep({
  number,
  title,
  description,
  status,
  isLast = false,
}: PipelineStepProps) {
  const style = statusStyles[status];

  return (
    <div className="flex items-start gap-4">
      {/* Step indicator */}
      <div className="flex flex-col items-center">
        <motion.div
          className={`w-10 h-10 rounded-full border-2 ${style.ring} ${style.bg}
                      flex items-center justify-center text-sm font-bold ${style.text}`}
          animate={status === 'active' ? { scale: [1, 1.1, 1] } : {}}
          transition={{ duration: 1.2, repeat: Infinity }}
        >
          {status === 'done' ? (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
            </svg>
          ) : (
            number
          )}
        </motion.div>
        {!isLast && (
          <div className={`w-px h-8 ${status === 'done' ? 'bg-green-300' : 'bg-slate-200'}`} />
        )}
      </div>

      {/* Step content */}
      <div className="pb-6">
        <h4 className={`font-medium text-sm ${style.text}`}>{title}</h4>
        <p className="text-slate-500 text-xs mt-0.5">{description}</p>
        {status === 'active' && (
          <motion.div
            className="mt-2 flex gap-1"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                className="w-1.5 h-1.5 rounded-full bg-blue-600"
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ duration: 1, delay: i * 0.2, repeat: Infinity }}
              />
            ))}
          </motion.div>
        )}
      </div>
    </div>
  );
}

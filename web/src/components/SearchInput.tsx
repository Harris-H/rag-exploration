'use client';

import { useState, type FormEvent } from 'react';

interface SearchInputProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  loading?: boolean;
}

export default function SearchInput({
  onSearch,
  placeholder = '输入搜索内容...',
  loading = false,
}: SearchInputProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed && !loading) {
      onSearch(trimmed);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-3 w-full max-w-2xl">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={placeholder}
        disabled={loading}
        className="flex-1 px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white
                   placeholder:text-white/30 focus:outline-none focus:border-blue-500/50
                   focus:ring-1 focus:ring-blue-500/30 transition-all disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={loading || !query.trim()}
        className="px-6 py-3 rounded-xl bg-blue-600 text-white font-medium
                   hover:bg-blue-500 active:bg-blue-700 transition-colors
                   disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
      >
        {loading ? (
          <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
          </svg>
        ) : (
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        )}
        搜索
      </button>
    </form>
  );
}

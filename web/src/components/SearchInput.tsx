'use client';

import { useState, type FormEvent } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

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
      <Input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={placeholder}
        disabled={loading}
        className="flex-1 h-11 px-4 rounded-xl"
      />
      <Button
        type="submit"
        disabled={loading || !query.trim()}
        className="h-11 px-6 rounded-xl bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white gap-2"
      >
        {loading ? (
          <Loader2 className="size-5 animate-spin" />
        ) : (
          <Search className="size-5" />
        )}
        搜索
      </Button>
    </form>
  );
}

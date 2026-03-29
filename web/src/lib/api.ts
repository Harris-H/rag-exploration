const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface BM25Result {
  title: string;
  content: string;
  score: number;
  tokens: string[];
}

export interface BM25Response {
  results: BM25Result[];
  query_tokens: string[];
  elapsed_ms: number;
}

export interface EmbeddingResult {
  title: string;
  content: string;
  score: number;
}

export interface EmbeddingResponse {
  results: EmbeddingResult[];
  embedding_dim: number;
  elapsed_ms: number;
}

export interface HybridResponse {
  bm25_results: EmbeddingResult[];
  embedding_results: EmbeddingResult[];
  hybrid_results: EmbeddingResult[];
  elapsed_ms: number;
}

export interface RAGResponse {
  rag_answer: string;
  direct_answer: string;
  retrieval_results: EmbeddingResult[];
  prompt: string;
  elapsed_ms: number;
}

export interface HealthResponse {
  status: string;
  documents_loaded: number;
  embedding_model: string;
  embedding_dim: number;
  ollama: {
    status: string;
    models: string[];
  };
}

async function post<T>(endpoint: string, body: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`API 请求失败 (${res.status}): ${text || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function bm25Search(query: string, topK: number = 5): Promise<BM25Response> {
  return post<BM25Response>('/api/bm25/search', { query, top_k: topK });
}

export async function embeddingSearch(query: string, topK: number = 5): Promise<EmbeddingResponse> {
  return post<EmbeddingResponse>('/api/embedding/search', { query, top_k: topK });
}

export async function hybridSearch(query: string, topK: number = 5): Promise<HybridResponse> {
  return post<HybridResponse>('/api/embedding/hybrid', { query, top_k: topK });
}

export async function ragQuery(query: string, topK: number = 3): Promise<RAGResponse> {
  return post<RAGResponse>('/api/rag/query', { query, top_k: topK, stream: false });
}

export function ragQueryStream(
  query: string,
  topK: number = 3,
  onEvent: (event: { type: string; data: unknown }) => void,
  onError: (error: Error) => void,
  onDone: () => void,
): AbortController {
  const controller = new AbortController();

  fetch(`${API_BASE}/api/rag/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK, stream: true }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`API 请求失败 (${res.status})`);
      const reader = res.body?.getReader();
      if (!reader) throw new Error('无法读取响应流');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              onEvent({ type: data.type || 'unknown', data });
            } catch {
              // skip malformed lines
            }
          }
        }
      }
      onDone();
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onError(err);
    });

  return controller;
}

export async function healthCheck(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`健康检查失败 (${res.status})`);
  return res.json() as Promise<HealthResponse>;
}

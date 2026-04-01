const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ModelsResponse {
  models: string[];
  default: string;
}

export async function fetchModels(): Promise<ModelsResponse> {
  const res = await fetch(`${API_BASE}/api/rag/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

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
      let currentEvent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            const rawData = line.slice(6);
            let parsedData: unknown = rawData;
            try {
              parsedData = JSON.parse(rawData);
            } catch {
              // data is plain text (e.g., tokens)
            }
            onEvent({ type: currentEvent || 'unknown', data: parsedData });
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

// --- Reranking types ---

export interface RerankResult {
  title: string;
  content: string;
  score: number;
  rank: number;
}

export interface RerankAfterResult extends RerankResult {
  original_rank: number;
  new_rank: number;
  rank_change: number;
}

export interface RerankResponse {
  before_rerank: RerankResult[];
  after_rerank: RerankAfterResult[];
  elapsed_ms: number;
  reranker_model: string;
}

export async function rerankSearch(query: string, topK: number = 5): Promise<RerankResponse> {
  return post<RerankResponse>('/api/reranking/search', { query, top_k: topK });
}

export async function healthCheck(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`健康检查失败 (${res.status})`);
  return res.json() as Promise<HealthResponse>;
}

// ── Evaluation Metrics ──────────────────────────────────────────────────

export interface MethodMetrics {
  mrr: number;
  precision_at_k: number;
  recall_at_k: number;
  ndcg_at_k: number;
}

export interface QueryEvalResult {
  query: string;
  description: string;
  relevant_docs: string[];
  methods: Record<string, {
    ranked_docs: string[];
    metrics: MethodMetrics;
  }>;
}

export interface EvalAllResponse {
  per_query: QueryEvalResult[];
  aggregate: Record<string, MethodMetrics>;
  k: number;
  num_queries: number;
  elapsed_ms: number;
}

export interface EvalQueryResponse {
  query: string;
  relevant_docs: string[];
  methods: Record<string, {
    ranked_docs: string[];
    metrics: MethodMetrics;
  }>;
  k: number;
  elapsed_ms: number;
}

export async function evaluateAll(k: number = 3): Promise<EvalAllResponse> {
  return post<EvalAllResponse>('/api/eval/all', { k });
}

export async function evaluateQuery(query: string, relevantDocIds: string[], k: number = 3): Promise<EvalQueryResponse> {
  return post<EvalQueryResponse>('/api/eval/query', { query, relevant_doc_ids: relevantDocIds, k });
}

// ── Chunking Strategies ──

export interface ChunkResult {
  text: string;
  score: number;
  full_length: number;
}

export interface StrategyResult {
  strategy_name: string;
  num_chunks: number;
  avg_chunk_len: number;
  min_chunk_len: number;
  max_chunk_len: number;
  top_chunks: ChunkResult[];
}

export interface ChunkingCompareResponse {
  strategies: Record<string, StrategyResult>;
  query: string;
  chunk_size: number;
  overlap: number;
  source_text_length: number;
  elapsed_ms: number;
}

export async function chunkingCompare(query: string, chunkSize: number = 150, overlap: number = 50): Promise<ChunkingCompareResponse> {
  return post<ChunkingCompareResponse>('/api/chunking/compare', { query, chunk_size: chunkSize, overlap });
}

// ── Enhanced RAG Pipeline ───────────────────────────────────────────────

export interface EnhancedRAGConfig {
  use_expansion: boolean;
  use_chunking: boolean;
  chunk_strategy: string | null;
  chunk_size: number | null;
  use_hybrid: boolean;
  use_reranking: boolean;
  top_k: number;
  model?: string;
}

export interface EnhancedChunkResult {
  text: string;
  doc_title: string;
  doc_id: string;
  score: number;
}

export interface EnhancedRerankResult extends EnhancedChunkResult {
  original_rank: number;
  new_rank: number;
  rank_change: number;
  rank?: number;
}

export interface EnhancedRerankData {
  before: EnhancedRerankResult[];
  after: EnhancedRerankResult[];
}

export interface QueryExpansionData {
  original: string;
  variants: string[];
  total_queries: number;
}

export interface ChunkingEventData {
  strategy: string;
  chunk_size: number;
  num_chunks: number;
  num_source_docs: number;
}

export interface CitationSource {
  index: number;
  doc_title: string;
  doc_id: string;
}

export interface EnhancedDoneData {
  full_answer: string;
  elapsed_ms: number;
  model: string;
  config: EnhancedRAGConfig;
  sources?: CitationSource[];
}

export function enhancedRagQueryStream(
  query: string,
  options: {
    topK?: number;
    useExpansion?: boolean;
    useChunking?: boolean;
    chunkStrategy?: string;
    chunkSize?: number;
    useHybrid?: boolean;
    useReranking?: boolean;
    model?: string;
  },
  onEvent: (event: { type: string; data: unknown }) => void,
  onError: (error: Error) => void,
  onDone: () => void,
): AbortController {
  const controller = new AbortController();

  const body = {
    query,
    top_k: options.topK ?? 3,
    use_expansion: options.useExpansion ?? false,
    use_chunking: options.useChunking ?? true,
    chunk_strategy: options.chunkStrategy ?? 'recursive',
    chunk_size: options.chunkSize ?? 200,
    use_hybrid: options.useHybrid ?? true,
    use_reranking: options.useReranking ?? true,
    model: options.model || null,
  };

  fetch(`${API_BASE}/api/rag/enhanced-query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error(`API 请求失败 (${res.status})`);
      const reader = res.body?.getReader();
      if (!reader) throw new Error('无法读取响应流');

      const decoder = new TextDecoder();
      let buffer = '';
      let currentEvent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            const rawData = line.slice(6);
            let parsedData: unknown = rawData;
            try {
              parsedData = JSON.parse(rawData);
            } catch {
              // plain text (e.g., tokens, prompt)
            }
            onEvent({ type: currentEvent || 'unknown', data: parsedData });
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

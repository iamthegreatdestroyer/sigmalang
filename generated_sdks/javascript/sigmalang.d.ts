/**
 * TypeScript definitions for the ΣLANG JavaScript SDK
 */

export interface EncodeOptions {
  strategy?: "auto" | "pattern" | "reference" | "delta" | "full";
  returnTree?: boolean;
}

export interface EncodeResult {
  encoded: string;
  ratio: number;
  strategy: string;
  tree?: object;
}

export interface DecodeResult {
  decoded: string;
  success: boolean;
}

export interface AnalogyOptions {
  topK?: number;
}

export interface AnalogyResult {
  answer: string;
  confidence: number;
  candidates: Array<{ text: string; score: number }>;
}

export interface SearchOptions {
  k?: number;
  threshold?: number;
}

export interface SearchResult {
  results: Array<{ id: string; text: string; score: number }>;
}

export interface Entity {
  text: string;
  type: string;
  start: number;
  end: number;
  confidence: number;
}

export interface Relation {
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
}

export interface SigmaLangClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  retries?: number;
}

export declare class SigmaLangError extends Error {
  statusCode: number | null;
  detail: any;
}

export declare class SigmaLangAPIError extends SigmaLangError {
  constructor(statusCode: number, body: any);
}

export declare class SigmaLangConnectionError extends SigmaLangError {
  cause: any;
}

export declare class SigmaLang {
  constructor(opts?: SigmaLangClientOptions);

  encode(text: string, opts?: EncodeOptions): Promise<EncodeResult>;
  decode(encoded: string): Promise<DecodeResult>;
  encodeBatch(texts: string[], opts?: EncodeOptions): Promise<EncodeResult[]>;
  roundTrip(
    text: string,
  ): Promise<{
    original: string;
    encoded: string;
    decoded: string;
    ratio: number;
  }>;

  analogy(
    a: string,
    b: string,
    c: string,
    opts?: AnalogyOptions,
  ): Promise<AnalogyResult>;
  analogyExplain(
    a: string,
    b: string,
  ): Promise<{ relationship: string; similarity: number; explanation: string }>;

  search(query: string, opts?: SearchOptions): Promise<SearchResult>;
  indexDocument(documentId: string, text: string): Promise<any>;

  extractEntities(text: string): Promise<{ entities: Entity[] }>;
  extractRelations(text: string): Promise<{ relations: Relation[] }>;

  embed(text: string): Promise<{ embedding: number[]; dimension: number }>;
  similarity(
    textA: string,
    textB: string,
  ): Promise<{ similarity: number; distance: number }>;
  embedBatch(texts: string[]): Promise<{ embeddings: number[][] }>;

  analyze(
    text: string,
  ): Promise<{
    ratio: number;
    entropy: number;
    strategy: string;
    timing_ms: number;
  }>;

  health(): Promise<{ status: string; version: string; uptime_s: number }>;
  status(): Promise<any>;
  version(): Promise<{
    version: string;
    python_version: string;
    build_date: string;
  }>;
  metrics(): Promise<string>;
}

export declare class SigmaLangStream {
  constructor(baseUrl?: string);
  encode(
    text: string,
    onChunk: (data: { chunk: string; encoded: string; done: boolean }) => void,
  ): Promise<void>;
}

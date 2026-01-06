/**
 * SigmaLang JavaScript SDK
 * ========================
 */

import axios, { AxiosInstance } from 'axios';

export interface CompressOptions {
  level?: 'fast' | 'balanced' | 'maximum';
  preserveFormatting?: boolean;
}

export interface AnalysisResult {
  compressionRatio: number;
  semanticDensity: number;
  complexity: number;
}

export class SigmaLang {
  private client: AxiosInstance;

  constructor(apiKey: string, baseURL: string = 'https://api.sigmalang.com') {
    this.client = axios.create({
      baseURL,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });
  }

  async compress(text: string, options?: CompressOptions): Promise<string> {
    const response = await this.client.post('/compress', {
      text,
      ...options
    });
    return response.data.compressed;
  }

  async decompress(compressed: string): Promise<string> {
    const response = await this.client.post('/decompress', {
      compressed
    });
    return response.data.text;
  }

  async analyze(text: string): Promise<AnalysisResult> {
    const response = await this.client.post('/analyze', { text });
    return response.data;
  }
}

export default SigmaLang;

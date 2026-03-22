const API_BASE_URL = 'http://localhost:8000/api';

export interface UploadResult {
  success: boolean;
  filename: string;
  chunks?: number;
  message?: string;
  error?: string;
}

export interface Source {
  filename: string;
  content: string;
  similarity: number;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  done?: boolean;
}

export interface Stats {
  total_documents: number;
  unique_files: number;
  embedding_model: string;
  llm_model: string;
  vector_dimension: number;
}

class ApiService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
        },
        ...options,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `请求失败: ${response.status}`);
      }

      return response.json();
    } catch (error) {
      console.error('API请求错误:', error);
      throw error;
    }
  }

  async uploadFiles(files: File[]): Promise<UploadResult[]> {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('上传文件失败');
    }

    const data = await response.json();
    return data.results;
  }

  async query(question: string, topK?: number): Promise<QueryResponse> {
    return this.request<QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify({ question, top_k: topK }),
    });
  }

  async *queryStream(question: string, topK?: number): AsyncGenerator<QueryResponse> {
    const response = await fetch(`${API_BASE_URL}/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question, top_k: topK }),
    });

    if (!response.ok) {
      throw new Error('查询失败');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('无法获取响应流');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim()) {
          try {
            const data = JSON.parse(line);
            yield data;
          } catch (e) {
            console.error('解析流数据失败:', e);
          }
        }
      }
    }
  }

  async getDocuments(): Promise<string[]> {
    const data = await this.request<{ documents: string[] }>('/documents');
    return data.documents;
  }

  async deleteDocument(filename: string): Promise<{ success: boolean; message: string }> {
    return this.request('/documents', {
      method: 'DELETE',
      body: JSON.stringify({ filename }),
    });
  }

  async getStats(): Promise<Stats> {
    return this.request<Stats>('/stats');
  }

  async clearKnowledgeBase(): Promise<{ success: boolean; message: string }> {
    return this.request('/clear', {
      method: 'DELETE',
    });
  }

  async healthCheck(): Promise<{ status: string; message: string }> {
    return this.request('/health');
  }
}

export const apiService = new ApiService();

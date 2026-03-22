import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import type { Stats } from '../services/api';

export const DocumentManager: React.FC = () => {
  const [documents, setDocuments] = useState<string[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [showClearModal, setShowClearModal] = useState(false);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    try {
      const [docs, statsData] = await Promise.all([
        apiService.getDocuments(),
        apiService.getStats(),
      ]);
      setDocuments(docs);
      setStats(statsData);
    } catch (error) {
      console.error('加载数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleDelete = async (filename: string) => {
    if (!confirm(`确定要删除文档 "${filename}" 吗？`)) {
      return;
    }

    setDeletingFile(filename);
    try {
      await apiService.deleteDocument(filename);
      await loadData();
    } catch (error) {
      console.error('删除文档失败:', error);
      alert('删除文档失败，请重试');
    } finally {
      setDeletingFile(null);
    }
  };

  const handleClearAll = async () => {
    try {
      await apiService.clearKnowledgeBase();
      setShowClearModal(false);
      await loadData();
    } catch (error) {
      console.error('清空知识库失败:', error);
      alert('清空知识库失败，请重试');
    }
  };

  if (loading) {
    return (<div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-center h-48">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
        </div>
      </div>);
  }

  return (<div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">文档管理</h3>
        {documents.length > 0 && (<button onClick={() => setShowClearModal(true)} className="px-3 py-1 text-sm text-red-600 border border-red-200 rounded-lg hover:bg-red-50 transition-colors">
            清空知识库
          </button>)}
      </div>

      {/* 统计信息 */}
      {stats && (<div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-blue-600">{stats.unique_files}</p>
            <p className="text-sm text-gray-600">文档数量</p>
          </div>
          <div className="bg-green-50 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-green-600">{stats.total_documents}</p>
            <p className="text-sm text-gray-600">文本块数量</p>
          </div>
          <div className="bg-purple-50 rounded-lg p-4 text-center">
            <p className="text-2xl font-bold text-purple-600">{stats.vector_dimension}</p>
            <p className="text-sm text-gray-600">向量维度</p>
          </div>
        </div>)}

      {/* 文档列表 */}
      <div className="space-y-2">
        {documents.length === 0 ? (<div className="text-center py-8 text-gray-500">
            <svg className="mx-auto h-12 w-12 text-gray-300 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
            </svg>
            <p className="text-sm">暂无文档</p>
            <p className="text-xs text-gray-400 mt-1">请先上传文档到知识库</p>
          </div>) : (documents.map((filename) => (<div key={filename} className="flex items-center justify-between bg-gray-50 rounded-lg p-3 hover:bg-gray-100 transition-colors">
              <div className="flex items-center space-x-3">
                <svg className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span className="text-sm text-gray-700">{filename}</span>
              </div>
              <button onClick={() => handleDelete(filename)} disabled={deletingFile === filename} className="text-gray-400 hover:text-red-500 disabled:opacity-50">
                {deletingFile === filename ? (<div className="animate-spin rounded-full h-4 w-4 border-b-2 border-red-500" />) : (<svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>)}
              </button>
            </div>)))}
      </div>

      {/* 模型信息 */}
      {stats && (<div className="mt-6 pt-4 border-t border-gray-200">
          <h4 className="text-sm font-medium text-gray-700 mb-2">模型信息</h4>
          <div className="space-y-1 text-xs text-gray-500">
            <p>嵌入模型: {stats.embedding_model}</p>
            <p>LLM模型: {stats.llm_model}</p>
          </div>
        </div>)}

      {/* 清空确认弹窗 */}
      {showClearModal && (<div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-md mx-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">
              确认清空知识库
            </h3>
            <p className="text-sm text-gray-600 mb-6">
              此操作将删除知识库中的所有文档，且无法恢复。确定要继续吗？
            </p>
            <div className="flex justify-end space-x-3">
              <button onClick={() => setShowClearModal(false)} className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                取消
              </button>
              <button onClick={handleClearAll} className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors">
                确认清空
              </button>
            </div>
          </div>
        </div>)}
    </div>);
};


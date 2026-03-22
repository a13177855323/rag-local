import React, { useState, useEffect } from 'react';
import { FileUpload } from './components/FileUpload';
import { ChatInterface } from './components/ChatInterface';
import { DocumentManager } from './components/DocumentManager';
import { apiService } from './services/api';

type TabType = 'chat' | 'upload' | 'documents';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('chat');
  const [isBackendOnline, setIsBackendOnline] = useState<boolean | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        await apiService.healthCheck();
        setIsBackendOnline(true);
      } catch (error) {
        setIsBackendOnline(false);
      }
    };

    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleUploadComplete = () => {
    setRefreshTrigger(prev => prev + 1);
    setActiveTab('chat');
  };

  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    {
      id: 'chat',
      label: '问答',
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
      ),
    },
    {
      id: 'upload',
      label: '上传',
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      ),
    },
    {
      id: 'documents',
      label: '文档',
      icon: (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
        </svg>
      ),
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 头部 */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <svg className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-800">本地 RAG 知识库</h1>
                <p className="text-xs text-gray-500">离线检索增强生成系统</p>
              </div>
            </div>

            {/* 后端状态 */}
            <div className="flex items-center space-x-2">
              <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm
                ${isBackendOnline ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}
              `}>
                <div className={`w-2 h-2 rounded-full
                  ${isBackendOnline ? 'bg-green-500' : 'bg-red-500 animate-pulse'}
                `} />
                <span>
                  {isBackendOnline === null
                    ? '连接中...'
                    : isBackendOnline
                    ? '后端在线'
                    : '后端离线'}
                </span>
              </div>
            </div>
          </div>

          {/* 标签页 */}
          <div className="mt-4 flex space-x-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors
                  ${activeTab === tab.id
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:bg-gray-100'}
                `}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* 主内容区 */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {isBackendOnline === false && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <svg className="h-5 w-5 text-red-500 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <div>
                <h3 className="text-sm font-medium text-red-800">后端服务未连接</h3>
                <p className="text-sm text-red-700 mt-1">
                  请确保后端服务正在运行。执行以下命令启动后端：
                </p>
                <code className="mt-2 block bg-red-100 px-3 py-2 rounded text-xs font-mono">
                  source venv/bin/activate && python -m backend.main
                </code>
              </div>
            </div>
          </div>
        )}

        <div className="h-[calc(100vh-220px)]">
          {activeTab === 'chat' && <ChatInterface />}
          {activeTab === 'upload' && (
            <div className="max-w-2xl mx-auto">
              <FileUpload onUploadComplete={handleUploadComplete} />
            </div>
          )}
          {activeTab === 'documents' && (
            <div key={refreshTrigger} className="max-w-2xl mx-auto">
              <DocumentManager />
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;

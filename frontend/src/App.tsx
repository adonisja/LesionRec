import { useState, useEffect } from 'react';
import { getCurrentUser, signOut } from 'aws-amplify/auth';
import { ImageUpload } from './components/ImageUpload';
import { AnalysisResults } from './components/AnalysisResults';
import { RecommendedProducts } from './components/RecommendedProducts';
import { Auth } from './components/Auth';
import type { AnalysisResponse } from './types';

type ViewState = 'dashboard' | 'upload' | 'results' | 'products';

function App() {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [currentView, setCurrentView] = useState<ViewState>('dashboard');
  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(null);

  useEffect(() => {
    checkUser();
    
    // Load saved analysis from local storage
    const saved = localStorage.getItem('lesionrec_last_analysis');
    if (saved) {
      try {
        setAnalysisData(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse saved analysis");
        localStorage.removeItem('lesionrec_last_analysis');
      }
    }
  }, []);

  async function checkUser() {
    try {
      const currentUser = await getCurrentUser();
      setUser(currentUser);
    } catch (err) {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleSignOut() {
    try {
      await signOut();
      setUser(null);
      setAnalysisData(null);
      localStorage.removeItem('lesionrec_last_analysis');
      setCurrentView('dashboard');
    } catch (error) {
      console.error('Error signing out: ', error);
    }
  }

  const handleAnalysisComplete = (data: AnalysisResponse) => {
    setAnalysisData(data);
    localStorage.setItem('lesionrec_last_analysis', JSON.stringify(data));
    setCurrentView('results');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!user) {
    return <Auth onLoginSuccess={checkUser} />;
  }

  return (
    <div className="min-h-screen bg-gray-100 py-12">
      <div className="max-w-4xl mx-auto mb-8 flex justify-between items-center px-6">
        <p className="text-gray-700 font-medium">
          Welcome back!
        </p>
        <button 
          onClick={handleSignOut}
          className="text-sm text-red-600 hover:text-red-800 font-medium"
        >
          Sign Out
        </button>
      </div>

      {/* Dashboard View */}
      {currentView === 'dashboard' && (
        <div className="max-w-md mx-auto text-center space-y-8">
          <h1 className="text-4xl font-bold text-gray-800">Skin Health Dashboard</h1>
          <p className="text-gray-600">Analyze your skin health with AI-powered insights.</p>
          
          {analysisData && (
            <button 
              onClick={() => setCurrentView('results')}
              className="w-full bg-white text-blue-600 border-2 border-blue-600 px-8 py-4 rounded-xl text-xl shadow-sm hover:bg-blue-50 transition-all mb-4"
            >
              View Last Analysis
            </button>
          )}

          <button 
            onClick={() => setCurrentView('upload')}
            className="w-full bg-blue-600 text-white px-8 py-4 rounded-xl text-xl shadow-lg hover:bg-blue-700 transition-all transform hover:scale-105"
          >
            Start New Analysis
          </button>
        </div>
      )}
      
      {/* Upload View */}
      {currentView === 'upload' && (
        <div className="max-w-md mx-auto">
          <button 
            onClick={() => setCurrentView('dashboard')}
            className="mb-4 text-gray-500 hover:text-gray-700 flex items-center"
          >
            ← Cancel
          </button>
          <ImageUpload 
            userId={user.userId} 
            onAnalysisComplete={handleAnalysisComplete} 
          />
        </div>
      )}

      {/* Results View */}
      {currentView === 'results' && analysisData && (
        <AnalysisResults 
          data={analysisData} 
          onBack={() => setCurrentView('dashboard')} 
          onViewProducts={() => setCurrentView('products')}
        />
      )}

      {/* Products View */}
      {currentView === 'products' && analysisData && (
        <RecommendedProducts 
          data={analysisData} 
          onBack={() => setCurrentView('results')} 
        />
      )}
    </div>
  );
}

export default App;

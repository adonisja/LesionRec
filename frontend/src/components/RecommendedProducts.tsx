import React, { useState } from 'react';
import type { AnalysisResponse } from '../types';
import { ProductRoutine } from './ProductRoutine';

interface Props {
    data: AnalysisResponse;
    onBack: () => void;
}

export const RecommendedProducts = ({ data, onBack }: Props) => {
    const [budget, setBudget] = useState<string>('');
    const [currentBundle, setCurrentBundle] = useState(data.product_recommendations.bundle || []);
    const [allRecommendations, setAllRecommendations] = useState(data.product_recommendations.recommendations || []);
    const [isUpdating, setIsUpdating] = useState(false);
    const [bundleStats, setBundleStats] = useState({
        totalCost: data.product_recommendations.total_cost || 0,
        savings: 0
    });
    const [lastUpdatedBudget, setLastUpdatedBudget] = useState<string | null>(null);

    // Automatically fetch bundle if it's empty but we have analysis
    React.useEffect(() => {
        if (currentBundle.length === 0 && data.ai_analysis.analysis) {
            handleUpdateBundle();
        }
    }, []);

    const handleUpdateBundle = async () => {
        setIsUpdating(true);
        try {
            const payload = {
                analysis_text: data.ai_analysis.analysis,
                budget_max: budget ? parseFloat(budget) : null
            };

            const response = await fetch('http://localhost:8000/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (response.ok) {
                const result = await response.json();
                setCurrentBundle(result.bundle);
                setAllRecommendations(result.recommendations || []);
                setBundleStats({
                    totalCost: result.total_cost,
                    savings: 0
                });
                setLastUpdatedBudget(budget || 'No Limit');
            } else {
                console.error("Failed to update bundle");
            }
        } catch (error) {
            console.error("Error updating bundle:", error);
        } finally {
            setIsUpdating(false);
        }
    };

    return (
        <div className="max-w-6xl mx-auto p-6 bg-white rounded-xl shadow-lg">
            <button onClick={onBack} className="mb-6 text-blue-600 hover:underline flex items-center font-medium">
                <span className="mr-2">←</span> Back to Analysis
            </button>

            <div className="border-b pb-6 mb-8">
                <h2 className="text-3xl font-bold text-gray-900">Recommended Products</h2>
                <p className="text-gray-500 mt-2">
                    Based on your skin analysis, we've curated this routine for you. 
                    Adjust your budget to see different options.
                </p>
            </div>

            <div className="flex flex-col md:flex-row justify-between items-end mb-8">
                <div className="text-center md:text-left w-full md:w-auto mb-4 md:mb-0">
                    <h3 className="text-xl font-semibold text-gray-800">Your Personalized Bundle</h3>
                </div>
                
                {/* Budget Control */}
                <div className="flex flex-col items-end">
                    <div className="flex items-center bg-gray-50 p-3 rounded-lg border border-gray-200 shadow-sm">
                        <label className="text-sm font-medium text-gray-700 mr-3">Max Budget:</label>
                        <div className="relative">
                            <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500">$</span>
                            <input 
                                type="number" 
                                value={budget}
                                onChange={(e) => setBudget(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleUpdateBundle()}
                                placeholder="No Limit"
                                className="pl-7 pr-3 py-1.5 w-28 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 text-right"
                                min="0"
                            />
                        </div>
                        <button 
                            onClick={handleUpdateBundle}
                            disabled={isUpdating}
                            className="ml-3 px-4 py-1.5 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 disabled:bg-blue-300 transition-colors"
                        >
                            {isUpdating ? 'Updating...' : 'Update Bundle'}
                        </button>
                    </div>
                    {lastUpdatedBudget !== null && (
                        <span className="text-xs text-gray-500 mt-1 mr-1">
                            Active Budget: <span className="font-medium">{lastUpdatedBudget === 'No Limit' ? 'No Limit' : `$${lastUpdatedBudget}`}</span>
                        </span>
                    )}
                </div>
            </div>

            {/* Bundle Summary */}
            {currentBundle.length > 0 && (
                <div className="mb-6 flex justify-end">
                    <div className="text-right">
                        <span className="text-gray-600 text-sm mr-2">Total Bundle Cost:</span>
                        <span className="text-xl font-bold text-green-600">${bundleStats.totalCost.toFixed(2)}</span>
                    </div>
                </div>
            )}
            
            {currentBundle.length > 0 ? (
                <div className="max-w-4xl mx-auto mb-12">
                    <ProductRoutine products={currentBundle} />
                </div>
            ) : (
                <div className="bg-gray-50 p-6 rounded-xl text-center text-gray-500 mb-12">
                    No specific product recommendations available for this budget. Try increasing it.
                </div>
            )}

            {/* Individual Recommendations Section */}
            {allRecommendations.length > 0 && (
                <div className="border-t pt-10">
                    <h3 className="text-2xl font-bold text-gray-900 mb-6">Individual Recommendations</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {allRecommendations.map((product: any, idx: number) => (
                            <div key={idx} className="border rounded-lg p-4 hover:shadow-md transition-shadow bg-white flex flex-col">
                                <div className="h-48 flex items-center justify-center mb-4 bg-gray-50 rounded-md overflow-hidden">
                                    {product.thumbnail ? (
                                        <img src={product.thumbnail} alt={product.title} className="max-h-full max-w-full object-contain" />
                                    ) : (
                                        <div className="text-gray-400">No Image</div>
                                    )}
                                </div>
                                <div className="flex-grow">
                                    <div className="text-xs font-bold text-blue-600 uppercase mb-1">{product.category || 'Product'}</div>
                                    <h4 className="text-sm font-medium text-gray-900 line-clamp-2 mb-2" title={product.title}>
                                        {product.title}
                                    </h4>
                                    <div className="flex items-center mb-2">
                                        <span className="text-yellow-400 mr-1">★</span>
                                        <span className="text-sm text-gray-600">{product.rating} ({product.reviews})</span>
                                    </div>
                                </div>
                                <div className="mt-4 flex items-center justify-between">
                                    <span className="text-lg font-bold text-gray-900">
                                        ${typeof product.price_numeric === 'number' ? product.price_numeric.toFixed(2) : product.price}
                                    </span>
                                    <a 
                                        href={product.link} 
                                        target="_blank" 
                                        rel="noopener noreferrer"
                                        className="px-3 py-1.5 bg-gray-900 text-white text-xs font-medium rounded hover:bg-gray-800"
                                    >
                                        View on Amazon
                                    </a>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

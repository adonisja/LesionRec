export interface AnalysisResponse {
    message: string;
    s3_path: string;
    ai_analysis: {
        analysis: string;
        // Add other fields as your backend expands
    };
    product_recommendations: {
        bundle?: any[];
        recommendations?: any[];
        total_cost?: number;
        budget_max?: number;
    };
}

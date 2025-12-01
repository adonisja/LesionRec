import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
from .product_data_cleaner import ProductDataCleaner


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ProductRecommender:
    """
    Match skin analysis to product recommendations
    
    Functions: 
        - recommend_for_condition: Get product recommendations for a specific condition
        - recommend_for_analysis: Extracts condition and severity from Gemini analysis and recommends products.
    """
    # Weights when customer prioritizes ratings
    RATING_PRIORITY_WEIGHTS = {
        'rating': 0.6,
        'reviews': 0.3,
        'price': 0.1
    }

    # Weights when customer prioritizes price
    PRICE_PRIORITY_WEIGHTS = {
        'rating': 0.4,
        'reviews': 0.2,
        'price': 0.4
    }
    
    
    def __init__(self):
        self.cleaner = ProductDataCleaner()
        self.products = self.cleaner.clean_all_products(save_cleaned=False)
        logger.info("Product data cleaned and loaded")


    def create_product_bundle(
        self,
        condition: str,
        budget_max: float = None,
        categories: List[str] = None,
        prioritize_rating: bool = True
    ):
        """
        Create a product bundle for a specific condition.

        Args:
            - condition: Skin condition to match (e.g. "acne")
            - budget_max: Maximum budget to filter price (default: None)
            - categories: List of product categories to filter
            - prioritize_rating: Weight rating higher than price (default: True)

        Returns:
            Dict with bundle products, total cost and savings
        """
        # Configure the weights depending on customer preference (price vs ratings)
        weights = self.RATING_PRIORITY_WEIGHTS if prioritize_rating else self.PRICE_PRIORITY_WEIGHTS

        if condition not in self.products:
            return {"error": "Condition not found", "bundle": []}
        
        df = self.products[condition].copy()

        if df.empty:
            return {f"error": "No products found for condition: {condition}", "bundle": []}
        
        # Categorize products by keywords in title
        df = self._categorize_products(df)

        # Default categories for skincare routine
        if categories is None:
            categories = ['cleanser', 'treatment', 'moisturizer']

        # Set up your knapsack and remaining space
        bundle = []
        remaining_budget = budget_max if budget_max is not None else float('inf')

        # Filter products based on category
        for category in categories:
            category_products = df[df['category'] == category]

            # Move to the next category if the product list is empty
            if category_products.empty:
                continue

            # Filter by remaining budget (use .copy() to avoid SettingWithCopyWarning)
            affordable = category_products[category_products['price_numeric'] <= remaining_budget].copy()

            # Move to the next category if there are no affordable items of that category
            # that can fit within the budget
            if affordable.empty:
                continue

            """ Calculate value score:
            Value scores are adjusted weights of the three most important columns depending on whether we should prioritize rating or not.
            """
            
            # Normalize rating to 0-1 scale so it's comparable to the normalized reviews score
            normalized_rating = affordable['rating'] / 5.0
            
            max_reviews = affordable['reviews'].max()
            reviews_score = (affordable['reviews'] / max_reviews) * weights['reviews'] if max_reviews > 0 else 0

            # If prioritizing rating, we treat price only as a constraint (must fit in budget),
            # not as a ranking factor. This allows higher-priced (but high-rated) items to be selected
            # if the budget allows.
            if prioritize_rating:
                price_score = 0
            else:
                max_price = affordable['price_numeric'].max()
                price_score = (affordable['price_numeric'] / max_price) * weights['price'] if max_price > 0 else 0

            affordable['value_score'] = (
                normalized_rating * weights['rating'] + reviews_score - price_score)
                
            

            # Select the best value product
            best_product_idx = affordable['value_score'].idxmax()
            best_product = affordable.loc[best_product_idx].to_dict()


            bundle.append({
                'category': category,
                'name': best_product['title'],
                'asin': best_product['asin'],
                'price': best_product['price'],
                'price_numeric': best_product['price_numeric'],
                'rating': best_product['rating'],
                'reviews': best_product['reviews'],
                'link': best_product['link'],
                'thumbnail': best_product['thumbnail'],
                'directions': best_product.get('directions', 'See product packaging for directions'),
                'value_score': best_product['value_score']
            })

            remaining_budget -= best_product['price_numeric']

        total_cost = sum(item['price_numeric'] for item in bundle)

        return {
            'bundle': bundle,
            'total_cost': total_cost,
            'budget_max': budget_max,
            'budget_utilized_pct': round((total_cost / budget_max) * 100, 2) if budget_max else 0,
            'categories_included': [item['category'] for item in bundle]
        }

    def _categorize_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize products by keywords in title.
        """
        category_keywords = {
            'cleanser': ['cleanser', 'face wash', 'wash', 'cleansing'],
            'treatment': ['treatment', 'serum', 'gel', 'spot treatment', 'adapalene', 'benzoyl', 'retinol'],
            'moisturizer': ['moisturizer', 'cream', 'lotion', 'hydrating'],
            'toner': ['toner', 'essence'],
            'sunscreen': ['sunscreen', 'spf', 'sun protection'],
            'mask': ['mask', 'peel'],
            'patch': ['patch', 'sticker']
        }

        def get_category(title):
            if pd.isna(title):
                return 'other'

            title_lower = title.lower()

            for category, keywords in category_keywords.items():
                if any(keyword in title_lower for keyword in keywords):
                    return category

            return 'other'

        df['category'] = df['title'].apply(get_category)
        return df

    def recommend_for_condition(
        self,
        condition: str,
        severity: str = 'moderate',
        budget_max: float = None,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        This function will get product recommendations for a specific condition
        based on facial analysis data.

        Args:
            - condition: Skin condition to match (e.g. "acne")
            - severity: Skin condition severity (expected: "mild"/"moderate"/"severe")
            - budget_max: Maximum budget to filter price
            - top_n: Number of products to return

        Returns:
            List of recommended products
        """

        if condition not in self.products:
            logger.warning(f"No products found for condition: {condition}")
            return []

        df = self.products[condition].copy()

        if df.empty:
            return []

        # Apply filters
        if budget_max is not None:
            df = df[df['price_numeric'] <= budget_max]

        # Sort by rating (primary) and reviews (secondary)
        df = df.sort_values(['rating','reviews'], ascending=[False, False])

        # Get top N products
        top_products = df.head(top_n)

        # Convert to list of dicts
        recommendations = top_products.to_dict('records')

        return recommendations

    def recommend_from_analysis(
        self,
        gemini_analysis: str,
        budget_max: float = None,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extracts condition and severity from Gemini analysis and recommends products.

        Args:
        gemini_analysis: Text from Gemini API
        budget_max: Maximum budget to filter price
        top_n: Number of products to return

        Returns:
        List of recommended products
        """
        condition, severity = self.extract_condition_and_severity(gemini_analysis)
        return self.recommend_for_condition(condition, severity, budget_max, top_n)

    def extract_condition_and_severity(self, gemini_analysis: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts condition and severity from Gemini JSON analysis.

        Now handles structured JSON output from Gemini 2.0 Flash.

        Args:
            gemini_analysis: JSON string from Gemini API

        Returns:
            Tuple of (condition, severity) or (None, None) if not found
        """
        try:
            # Parse JSON response
            analysis_data = json.loads(gemini_analysis)

            # Extract severity directly from JSON (already normalized)
            severity = analysis_data.get('severity', '').lower()
            if severity not in ['mild', 'moderate', 'severe']:
                logger.warning(f"Invalid severity '{severity}', defaulting to moderate")
                severity = 'moderate'

            # Extract characterization and search for condition
            characterization = analysis_data.get('characterization', '').lower()

            condition_keywords = {
                'acne': ['acne', 'pimple', 'blemish', 'breakout', 'comedone', 'whitehead', 'blackhead', 'papule', 'pustule'],
                'rosacea': ['rosacea', 'facial redness', 'flushing', 'telangiectasia'],
                'eczema': ['eczema', 'atopic dermatitis', 'dry patches', 'dermatitis'],
                'dry_skin': ['dry skin', 'dehydrated', 'flaky', 'xerosis'],
                'oily_skin': ['oily', 'sebum', 'shine', 'sebaceous'],
                'hyperpigmentation': ['hyperpigmentation', 'dark spots', 'discoloration', 'melasma', 'post-inflammatory'],
                'melasma': ['melasma', 'brown patches', 'chloasma']
            }

            # Search characterization for condition keywords
            detected_condition = None
            for condition, keywords in condition_keywords.items():
                if any(keyword in characterization for keyword in keywords):
                    detected_condition = condition
                    logger.info(f"Detected condition: {condition} (severity: {severity})")
                    break

            if not detected_condition:
                logger.warning(f"No condition detected in: {characterization[:100]}")
                return None, None

            return detected_condition, severity

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            logger.error(f"Raw response: {gemini_analysis[:200]}")

            # Fallback: try keyword search on raw text (old method)
            return self._extract_condition_from_text_fallback(gemini_analysis)

        except Exception as e:
            logger.error(f"Unexpected error extracting condition: {e}")
            return None, None

    def _extract_condition_from_text_fallback(self, text: str) -> Tuple[Optional[str], str]:
        """
        Fallback method for parsing unstructured text responses.
        Used when JSON parsing fails.
        """
        logger.warning("Using fallback text parsing (JSON parse failed)")

        text_lower = text.lower()

        condition_keywords = {
            'acne': ['acne', 'pimple', 'blemish', 'breakout', 'comedone', 'whitehead', 'blackhead'],
            'rosacea': ['rosacea', 'facial redness', 'flushing'],
            'eczema': ['eczema', 'atopic dermatitis', 'dry patches'],
            'dry_skin': ['dry skin', 'dehydrated', 'flaky'],
            'oily_skin': ['oily', 'sebum', 'shine'],
            'hyperpigmentation': ['hyperpigmentation', 'dark spots', 'discoloration'],
            'melasma': ['melasma', 'brown patches']
        }

        severity_keywords = {
            'mild': ['mild', 'light', 'minor'],
            'moderate': ['moderate', 'medium'],
            'severe': ['severe', 'serious', 'heavy', 'extensive']
        }

        # Find condition
        detected_condition = None
        for condition, keywords in condition_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_condition = condition
                break

        # Find severity
        detected_severity = 'moderate'  # default
        for severity_level, keywords in severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_severity = severity_level
                break

        if detected_condition:
            logger.info(f"Fallback detected: {detected_condition} ({detected_severity})")

        return detected_condition, detected_severity

    def create_product_bundle_from_analysis(
        self,
        gemini_analysis: str,
        budget_max: float = None,
        categories: List[str] = None,
        prioritize_rating: bool = True
    ) -> Dict[str, Any]:
        """
        Create a product bundle based on Gemini analysis results.

        Args:
            gemini_analysis: Text from Gemini API
            budget_max: Maximum budget for the bundle (None = Infinite)
            categories: List of product categories (default: cleanser, treatment, moisturizer)
            prioritize_rating: Weight rating higher than price

        Returns:
            Dict with bundle products, individual recommendations, and metadata
        """
        condition, severity = self.extract_condition_and_severity(gemini_analysis)

        if condition is None:
            return {
                "error": "Could not detect skin condition from analysis",
                "bundle": [],
                "recommendations": [],
                "total_cost": 0
            }

        # 1. Get the Bundle (Sum <= Budget)
        bundle_result = self.create_product_bundle(
            condition=condition,
            budget_max=budget_max,
            categories=categories,
            prioritize_rating=prioritize_rating
        )
        
        # 2. Get Individual Recommendations (Item Price <= Budget)
        # If budget_max is None, it returns everything (top N).
        # If budget_max is set, it filters items > budget_max.
        individual_recommendations = self.recommend_for_condition(
            condition=condition,
            severity=severity,
            budget_max=budget_max,
            top_n=10 # Get more items for the list
        )
        
        # Merge results
        bundle_result['recommendations'] = individual_recommendations
        
        return bundle_result


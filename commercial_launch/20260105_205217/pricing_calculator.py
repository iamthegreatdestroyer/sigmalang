"""
SigmaLang Pricing Calculator
========================

Dynamic pricing calculation based on usage and value metrics.
"""

from typing import Dict, Any
import json

class PricingCalculator:
    """Dynamic pricing calculator for SigmaLang"""

    def __init__(self, pricing_model_path: str):
        with open(pricing_model_path, 'r') as f:
            self.model = json.load(f)

    def calculate_price(self, usage_metrics: Dict[str, Any], demand_level: str = "normal") -> Dict[str, Any]:
        """Calculate optimal price based on usage metrics"""
        # Calculate value score
        value_score = self._calculate_value_score(usage_metrics)

        # Determine base tier
        base_tier = self._determine_base_tier(usage_metrics)

        # Apply dynamic pricing
        demand_multiplier = self.model["dynamic_pricing"]["demand_multiplier"][demand_level]

        # Calculate volume discount
        volume_discount = self._calculate_volume_discount(usage_metrics)

        # Final price calculation
        base_price = self.model["pricing_tiers"][base_tier]["monthly_price"]
        final_price = base_price * demand_multiplier * (1 - volume_discount)

        return {
            "recommended_tier": base_tier,
            "base_price": base_price,
            "final_price": round(final_price, 2),
            "demand_multiplier": demand_multiplier,
            "volume_discount": volume_discount,
            "value_score": value_score
        }

    def _calculate_value_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate value score from metrics"""
        score = 0
        weights = self.model["base_metrics"]

        for metric, config in weights.items():
            if metric in metrics:
                value = metrics[metric]
                baseline = config["baseline"]
                premium = config["premium_threshold"]
                weight = config["weight"]

                # Normalize to 0-1 scale
                if value >= premium:
                    normalized = 1.0
                elif value <= baseline:
                    normalized = 0.0
                else:
                    normalized = (value - baseline) / (premium - baseline)

                score += normalized * weight

        return min(score, 1.0)  # Cap at 1.0

    def _determine_base_tier(self, metrics: Dict[str, Any]) -> str:
        """Determine appropriate base tier"""
        requests_per_month = metrics.get("requests_per_month", 0)

        if requests_per_month <= 1000000:
            return "starter"
        elif requests_per_month <= 10000000:
            return "professional"
        else:
            return "enterprise"

    def _calculate_volume_discount(self, metrics: Dict[str, Any]) -> float:
        """Calculate volume-based discount"""
        requests_per_month = metrics.get("requests_per_month", 0)

        for discount_rule in self.model["dynamic_pricing"]["volume_discounts"]:
            if requests_per_month >= discount_rule["threshold"]:
                return discount_rule["discount"]

        return 0.0

# Example usage
if __name__ == "__main__":
    calculator = PricingCalculator("pricing_model.json")

    # Example usage metrics
    metrics = {
        "compression_ratio": 30.0,  # 30x compression
        "processing_speed": 5,      # 5ms response time
        "accuracy": 0.995,          # 99.5% accuracy
        "requests_per_month": 5000000
    }

    price = calculator.calculate_price(metrics, demand_level="high")
    print(f"Recommended pricing: ${price['final_price']}/month (Tier: {price['recommended_tier']})")

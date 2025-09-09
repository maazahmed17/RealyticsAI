#!/usr/bin/env python3
"""
Comprehensive Test Suite for Price Prediction Model
====================================================
Tests location sensitivity, feature sensitivity, BHK logic, 
consistency, and market reasonableness.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from backend.services.price_prediction.enhanced_price_predictor import EnhancedPricePredictionService

console = Console()


class PricePredictionTester:
    """Comprehensive test suite for price prediction model"""
    
    def __init__(self):
        self.service = EnhancedPricePredictionService()
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all test suites"""
        console.print("\n[bold cyan]🧪 RUNNING COMPREHENSIVE PRICE PREDICTION TESTS[/bold cyan]")
        console.print("=" * 80)
        
        # Run each test suite
        await self.test_location_sensitivity()
        await self.test_feature_sensitivity()
        await self.test_bhk_logic()
        await self.test_prediction_consistency()
        await self.test_market_reasonableness()
        
        # Display summary
        self.display_test_summary()
    
    async def test_location_sensitivity(self):
        """Test 1: Location should significantly affect price"""
        console.print("\n[yellow]📍 TEST 1: Location Sensitivity[/yellow]")
        
        # Test properties with same specs but different locations
        test_cases = [
            {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 1650, "location": "Koramangala"},
            {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 1650, "location": "Hebbal"},
            {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 1650, "location": "Whitefield"},
            {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 1650, "location": "Electronic City"},
            {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 1650, "location": "Yelahanka"}
        ]
        
        results = []
        for features in test_cases:
            result = await self.service.predict_price(features)
            results.append({
                "location": features["location"],
                "price": result["predicted_price"]
            })
        
        # Analyze results
        prices = [r["price"] for r in results]
        price_variance = max(prices) - min(prices)
        variance_pct = (price_variance / min(prices)) * 100
        
        # Check if locations have different prices
        unique_prices = len(set(prices))
        passed = unique_prices > 1 and variance_pct > 10
        
        # Display results
        table = Table(title="Location Sensitivity Test Results")
        table.add_column("Location", style="cyan")
        table.add_column("Price (Lakhs)", style="yellow")
        
        for r in results:
            table.add_row(r["location"], f"₹{r['price']:.2f}")
        
        console.print(table)
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        console.print(f"\nPrice Variance: {variance_pct:.1f}%")
        console.print(f"Unique Prices: {unique_prices}/{len(results)}")
        console.print(f"Test Status: {status}")
        
        self.test_results.append({
            "test": "Location Sensitivity",
            "passed": passed,
            "details": f"Variance: {variance_pct:.1f}%, Unique: {unique_prices}"
        })
    
    async def test_feature_sensitivity(self):
        """Test 2: Features like sqft should affect price"""
        console.print("\n[yellow]📐 TEST 2: Feature Sensitivity (Square Footage)[/yellow]")
        
        # Test properties with different square footage
        test_cases = [
            {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 1200, "location": "Hebbal"},
            {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 1500, "location": "Hebbal"},
            {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 1800, "location": "Hebbal"},
            {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 2100, "location": "Hebbal"},
        ]
        
        results = []
        for features in test_cases:
            result = await self.service.predict_price(features)
            results.append({
                "sqft": features["total_sqft"],
                "price": result["predicted_price"]
            })
        
        # Check if price increases with sqft
        price_increases = all(
            results[i]["price"] <= results[i+1]["price"] 
            for i in range(len(results)-1)
        )
        
        # Calculate price per sqft change
        price_changes = []
        for i in range(1, len(results)):
            sqft_diff = results[i]["sqft"] - results[i-1]["sqft"]
            price_diff = results[i]["price"] - results[i-1]["price"]
            change_pct = (price_diff / results[i-1]["price"]) * 100
            price_changes.append(change_pct)
        
        avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
        passed = price_increases and avg_change > 3
        
        # Display results
        table = Table(title="Square Footage Sensitivity Results")
        table.add_column("Sqft", style="cyan")
        table.add_column("Price (Lakhs)", style="yellow")
        table.add_column("Change %", style="green")
        
        for i, r in enumerate(results):
            change = f"+{price_changes[i-1]:.1f}%" if i > 0 else "-"
            table.add_row(str(r["sqft"]), f"₹{r['price']:.2f}", change)
        
        console.print(table)
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        console.print(f"\nPrice increases with sqft: {price_increases}")
        console.print(f"Average price change: {avg_change:.1f}%")
        console.print(f"Test Status: {status}")
        
        self.test_results.append({
            "test": "Feature Sensitivity",
            "passed": passed,
            "details": f"Monotonic: {price_increases}, Avg Change: {avg_change:.1f}%"
        })
    
    async def test_bhk_logic(self):
        """Test 3: BHK should significantly affect price"""
        console.print("\n[yellow]🏠 TEST 3: BHK Logic Test[/yellow]")
        
        # Test properties with different BHK
        test_cases = [
            {"bhk": 1, "bath": 1, "balcony": 1, "total_sqft": 650, "location": "Marathahalli"},
            {"bhk": 2, "bath": 2, "balcony": 1, "total_sqft": 1100, "location": "Marathahalli"},
            {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 1650, "location": "Marathahalli"},
            {"bhk": 4, "bath": 3, "balcony": 2, "total_sqft": 2200, "location": "Marathahalli"},
        ]
        
        results = []
        for features in test_cases:
            result = await self.service.predict_price(features)
            results.append({
                "bhk": features["bhk"],
                "price": result["predicted_price"]
            })
        
        # Check if price increases with BHK
        price_increases = all(
            results[i]["price"] < results[i+1]["price"] * 1.5  # Allow some margin
            for i in range(len(results)-1)
        )
        
        # Calculate price ratios
        price_ratios = []
        for i in range(1, len(results)):
            ratio = results[i]["price"] / results[0]["price"]
            price_ratios.append(ratio)
        
        # 3BHK should be at least 20% more than 1BHK
        bhk3_vs_bhk1 = (results[2]["price"] / results[0]["price"] - 1) * 100 if len(results) > 2 else 0
        passed = price_increases and bhk3_vs_bhk1 > 20
        
        # Display results
        table = Table(title="BHK Logic Test Results")
        table.add_column("BHK", style="cyan")
        table.add_column("Price (Lakhs)", style="yellow")
        table.add_column("Ratio to 1BHK", style="green")
        
        for i, r in enumerate(results):
            ratio = f"{results[i]['price']/results[0]['price']:.2f}x" if i > 0 else "1.00x"
            table.add_row(f"{r['bhk']} BHK", f"₹{r['price']:.2f}", ratio)
        
        console.print(table)
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        console.print(f"\n3BHK vs 1BHK difference: +{bhk3_vs_bhk1:.1f}%")
        console.print(f"Test Status: {status}")
        
        self.test_results.append({
            "test": "BHK Logic",
            "passed": passed,
            "details": f"3BHK vs 1BHK: +{bhk3_vs_bhk1:.1f}%"
        })
    
    async def test_prediction_consistency(self):
        """Test 4: Same input should give same output"""
        console.print("\n[yellow]🔄 TEST 4: Prediction Consistency[/yellow]")
        
        # Test same property multiple times
        features = {
            "bhk": 3,
            "bath": 2,
            "balcony": 2,
            "total_sqft": 1500,
            "location": "Indiranagar"
        }
        
        predictions = []
        for i in range(5):
            result = await self.service.predict_price(features)
            predictions.append(result["predicted_price"])
        
        # Check consistency
        unique_predictions = len(set(predictions))
        all_same = unique_predictions == 1
        
        # Display results
        table = Table(title="Consistency Test Results")
        table.add_column("Run #", style="cyan")
        table.add_column("Price (Lakhs)", style="yellow")
        
        for i, price in enumerate(predictions, 1):
            table.add_row(f"Run {i}", f"₹{price:.2f}")
        
        console.print(table)
        
        status = "✅ PASSED" if all_same else "❌ FAILED"
        console.print(f"\nUnique predictions: {unique_predictions}")
        console.print(f"Test Status: {status}")
        
        self.test_results.append({
            "test": "Consistency",
            "passed": all_same,
            "details": f"Unique outputs: {unique_predictions}/5"
        })
    
    async def test_market_reasonableness(self):
        """Test 5: Predictions should be within reasonable market bounds"""
        console.print("\n[yellow]💰 TEST 5: Market Reasonableness[/yellow]")
        
        test_cases = [
            {
                "features": {"bhk": 1, "bath": 1, "balcony": 0, "total_sqft": 500, "location": "Electronic City"},
                "expected_range": (20, 60)  # 1BHK in budget area
            },
            {
                "features": {"bhk": 3, "bath": 2, "balcony": 2, "total_sqft": 1650, "location": "Koramangala"},
                "expected_range": (100, 250)  # 3BHK in premium area
            },
            {
                "features": {"bhk": 4, "bath": 3, "balcony": 2, "total_sqft": 2500, "location": "Whitefield"},
                "expected_range": (120, 300)  # 4BHK in IT hub
            },
            {
                "features": {"bhk": 2, "bath": 2, "balcony": 1, "total_sqft": 1000, "location": "Hebbal"},
                "expected_range": (50, 120)  # 2BHK in mid-tier area
            }
        ]
        
        results = []
        all_reasonable = True
        
        for test in test_cases:
            result = await self.service.predict_price(test["features"])
            price = result["predicted_price"]
            min_price, max_price = test["expected_range"]
            is_reasonable = min_price <= price <= max_price
            
            if not is_reasonable:
                all_reasonable = False
            
            results.append({
                "property": f"{test['features']['bhk']}BHK in {test['features']['location']}",
                "price": price,
                "expected": f"₹{min_price}-{max_price}L",
                "reasonable": is_reasonable
            })
        
        # Display results
        table = Table(title="Market Reasonableness Test Results")
        table.add_column("Property", style="cyan")
        table.add_column("Predicted", style="yellow")
        table.add_column("Expected Range", style="green")
        table.add_column("Status", style="magenta")
        
        for r in results:
            status = "✅" if r["reasonable"] else "❌"
            table.add_row(
                r["property"],
                f"₹{r['price']:.2f}L",
                r["expected"],
                status
            )
        
        console.print(table)
        
        passed = all_reasonable
        status = "✅ PASSED" if passed else "❌ FAILED"
        console.print(f"\nTest Status: {status}")
        
        self.test_results.append({
            "test": "Market Reasonableness",
            "passed": passed,
            "details": f"All within bounds: {all_reasonable}"
        })
    
    def display_test_summary(self):
        """Display summary of all test results"""
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]📊 TEST SUMMARY[/bold cyan]")
        
        # Create summary table
        table = Table(title="Final Test Results", show_header=True)
        table.add_column("Test Suite", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Details", style="white")
        
        passed_count = 0
        for result in self.test_results:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            if result["passed"]:
                passed_count += 1
            table.add_row(result["test"], status, result["details"])
        
        console.print(table)
        
        # Overall result
        total_tests = len(self.test_results)
        success_rate = (passed_count / total_tests) * 100 if total_tests > 0 else 0
        
        if success_rate == 100:
            overall = "[bold green]🎉 ALL TESTS PASSED![/bold green]"
            console.print(Panel(overall, title="Overall Result", border_style="green"))
        elif success_rate >= 80:
            overall = f"[yellow]⚠️ MOSTLY PASSED ({passed_count}/{total_tests})[/yellow]"
            console.print(Panel(overall, title="Overall Result", border_style="yellow"))
        else:
            overall = f"[red]❌ NEEDS IMPROVEMENT ({passed_count}/{total_tests})[/red]"
            console.print(Panel(overall, title="Overall Result", border_style="red"))
        
        console.print(f"\nSuccess Rate: {success_rate:.1f}%")
        console.print("=" * 80)


async def main():
    """Main test runner"""
    tester = PricePredictionTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

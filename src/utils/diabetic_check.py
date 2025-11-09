# src/utils/diabetic_check.py
"""
Utility function to determine if a food item is safe for diabetics
based on Glycemic Index (GI) and Glycemic Load (GL).
"""


def analyze_food_for_diabetics(name, glycemic_index, carbs_per_serving):
    """
    Analyze a food item for diabetic safety based on GI and GL.
    
    Args:
        name (str): Food name
        glycemic_index (float): Glycemic Index value
        carbs_per_serving (float): Carbohydrates per serving in grams
    
    Returns:
        dict: Analysis result with name, GI, GL, and classification
    """
    # Calculate Glycemic Load
    glycemic_load = (glycemic_index * carbs_per_serving) / 100
    
    # Determine classification (check in order: Avoid → Caution → Safe)
    if glycemic_index >= 70 or glycemic_load >= 20:
        classification = "High GI/GL — Avoid ❌"
    elif glycemic_index <= 55 and glycemic_load <= 10:
        classification = "Low GI/GL — Safe for diabetics ✅"
    else:
        # GI 56-69 or GL 11-19 (but not both in avoid range)
        classification = "Medium GI/GL — Use with caution ⚠️"
    
    return {
        "name": name,
        "glycemicIndex": glycemic_index,
        "glycemicLoad": round(glycemic_load, 2),
        "classification": classification
    }


# Sample foods with GI and carbs values
SAMPLE_FOODS = [
    {"name": "Apple", "glycemicIndex": 40, "carbsPerServing": 20},
    {"name": "White Rice", "glycemicIndex": 73, "carbsPerServing": 45},
    {"name": "Brown Rice", "glycemicIndex": 68, "carbsPerServing": 45},
    {"name": "Banana", "glycemicIndex": 51, "carbsPerServing": 27},
    {"name": "Watermelon", "glycemicIndex": 76, "carbsPerServing": 12},
    {"name": "Oatmeal", "glycemicIndex": 55, "carbsPerServing": 30},
    {"name": "Sweet Potato", "glycemicIndex": 70, "carbsPerServing": 20},
    {"name": "Quinoa", "glycemicIndex": 53, "carbsPerServing": 39},
    {"name": "White Bread", "glycemicIndex": 75, "carbsPerServing": 15},
    {"name": "Lentils", "glycemicIndex": 32, "carbsPerServing": 20},
]


def display_diabetic_safety_ratings(foods=None):
    """
    Display diabetic safety ratings for a list of foods.
    
    Args:
        foods (list, optional): List of food dictionaries with 'name', 'glycemicIndex', 
                               and 'carbsPerServing'. If None, uses SAMPLE_FOODS.
    """
    if foods is None:
        foods = SAMPLE_FOODS
    
    print("\n" + "="*70)
    print("DIABETIC SAFETY ANALYSIS")
    print("="*70)
    print(f"{'Food':<20} {'GI':<8} {'Carbs':<10} {'GL':<8} {'Classification':<35}")
    print("-"*70)
    
    for food in foods:
        result = analyze_food_for_diabetics(
            food["name"],
            food["glycemicIndex"],
            food["carbsPerServing"]
        )
        print(f"{result['name']:<20} {result['glycemicIndex']:<8} "
              f"{food['carbsPerServing']:<10} {result['glycemicLoad']:<8} "
              f"{result['classification']:<35}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Example 1: Single food analysis")
    print("-" * 50)
    result = analyze_food_for_diabetics("Apple", 40, 20)
    print(f"Name: {result['name']}")
    print(f"Glycemic Index: {result['glycemicIndex']}")
    print(f"Glycemic Load: {result['glycemicLoad']}")
    print(f"Classification: {result['classification']}")
    
    # Display ratings for sample foods
    display_diabetic_safety_ratings()


"""
Health Knowledge Base for Carbs-AI (6 supported classes)
"""

def get_gi_analysis(gi):
    if gi <= 55:
        return {
            "category": "Low GI",
            "metabolic_impact": "Slow digestion, gradual rise in blood sugar.",
            "diabetes_guidance": "Preferred choice. Helps maintain stable glucose levels.",
            "risk_color": "green"
        }
    elif 56 <= gi <= 69:
        return {
            "category": "Medium GI",
            "metabolic_impact": "Moderate digestion speed. Can cause spikes if eaten alone.",
            "diabetes_guidance": "Consume in moderation. Pair with fiber/protein to lower impact.",
            "risk_color": "yellow"
        }
    else:
        return {
            "category": "High GI",
            "metabolic_impact": "Rapid digestion. Causes sharp blood sugar spikes and insulin surges.",
            "diabetes_guidance": "Avoid or strictly limit. Risk of hyperglycemia.",
            "risk_color": "red"
        }

FOOD_KNOWLEDGE = {
    "biryani": {
        "kidney_safety": "Moderate Risk. Often high in sodium from spices/salt and protein from meat. Potassium varies with veggies.",
        "conditions": {
            "PCOS": "Limit portion. White rice base can spike insulin, worsening symptoms.",
            "Obesity": "High Calorie density. Save for occasional treats.",
            "Heart": "High Sodium/Fat risk. Watch for saturated fats (ghee).",
            "Hypertension": "Caution: Salt content is usually high.",
            "Cholesterol": "Caution if made with red meat/ghee.",
            "Fatty Liver": "Limit intake due to high carb+fat combination.",
            "Gut Health": "Spices may trigger IBS/Reflux in sensitive individuals."
        },
        "recommendation": "Occasional/Treat",
        "portion_size": "1 cup (150g)"
    },
    "dal": {
        "kidney_safety": "Monitor Protein/Potassium. Lentils are high in potassium and protein. CKD patients may need leeching.",
        "conditions": {
            "PCOS": "Excellent choice. High fiber/protein helps manage insulin.",
            "Obesity": "Good for weight loss (high satiety). Avoid excess tadka (oil).",
            "Heart": "Heart healthy (fiber lowers cholesterol).",
            "Hypertension": "Good (DASH diet friendly) if low salt.",
            "Cholesterol": "Helps lower LDL.",
            "Fatty Liver": "Beneficial (Choline/Fiber).",
            "Gut Health": "High fiber. Can cause bloating (gas) in IBS."
        },
        "recommendation": "Regular/Staple",
        "portion_size": "1 bowl (200g)"
    },
    "halwa": {
        "kidney_safety": "Low Protein, but check Potassium (if nuts/carrots user). High Sugar is the main kidney stressor long-term.",
        "conditions": {
            "PCOS": "Avoid. High sugar worsen's insulin resistance.",
            "Obesity": "Avoid. Calorie dense, low satiety.",
            "Heart": "High saturated fat (ghee) and sugar. Inflammatory.",
            "Hypertension": "Neutral short-term, bad long-term (metabolic syndrome).",
            "Cholesterol": "Raises LDL/Triglycerides.",
            "Fatty Liver": "High Risk (Fructose/Sugar drives liver fat).",
            "Gut Health": "High sugar can feed bad gut bacteria."
        },
        "recommendation": "Avoid/Rare Treat",
        "portion_size": "2-3 tbsp (50g)"
    },
    "poha": {
        "kidney_safety": "Generally Safe. Low sodium (if homemade), moderate potassium. Easy to digest.",
        "conditions": {
            "PCOS": "Moderate. It is flattened rice (High GI). Add peanuts/veggies to lower impact.",
            "Obesity": "Good breakfast if portion controlled. Light but not very satiating alone.",
            "Heart": "Good (low fat).",
            "Hypertension": "Safe if low salt.",
            "Cholesterol": "Neutral.",
            "Fatty Liver": "Safe in moderation.",
            "Gut Health": "Easily digestible. Good for sensitive stomachs."
        },
        "recommendation": "Regular (with veggies)",
        "portion_size": "1.5 cups"
    },
    "rasgulla": {
        "kidney_safety": "Good for low-protein diets (if made from milk solids, has some protein). High sugar is the concern.",
        "conditions": {
            "PCOS": "Avoid. Pure sugar syrup.",
            "Obesity": "Avoid. Empty calories.",
            "Heart": "Neutral immediate risk, bad metabolic risk.",
            "Hypertension": "Neutral.",
            "Cholesterol": "Neutral.",
            "Fatty Liver": "High Risk (Sugar).",
            "Gut Health": "Dairy (Chenna) triggers lactose intolerance."
        },
        "recommendation": "Avoid/Rare Treat",
        "portion_size": "1 piece"
    },
    "roti": {
        "kidney_safety": "Moderate Potassium (Whole wheat). White flour is lower potassium but less healthy.",
        "conditions": {
            "PCOS": "Choose Multigrain. Good complex carb.",
            "Obesity": "Limit quantity. Calorie dense.",
            "Heart": "Good (Whole grains).",
            "Hypertension": "Safe.",
            "Cholesterol": "Good (Fiber).",
            "Fatty Liver": "Safe staple.",
            "Gut Health": "Contains Gluten. Watch for Celiac/NCGS."
        },
        "recommendation": "Regular/Staple",
        "portion_size": "2 medium rotis"
    }
}

def analyze_food_health(dish_name, gi):
    data = FOOD_KNOWLEDGE.get(dish_name.lower(), {})
    gi_analysis = get_gi_analysis(gi)
    
    return {
        "dish": dish_name,
        "gi_analysis": gi_analysis,
        "specifics": data
    }

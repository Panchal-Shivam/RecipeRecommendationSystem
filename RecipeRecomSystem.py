import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    """Preprocesses text data."""
    if isinstance(text, str):
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.strip()  # Strip leading and trailing spaces
    else:
        return ''  # Return empty string for non-string inputs

def calculate_similarity(user_ingredients, recipe_ingredients, vectorizer):
    """Calculates cosine similarity between user input and recipe ingredients."""
    user_vector = vectorizer.transform([user_ingredients])
    recipe_vector = vectorizer.transform([recipe_ingredients])
    similarity_score = cosine_similarity(user_vector, recipe_vector)[0][0]
    return similarity_score

def recommend_recipes(user_ingredients, recipe_data, vectorizer):
    """Recommends recipes based on user input."""
    priority_recipes = []
    other_recipes = []

    user_ingredients_list = user_ingredients.split(", ")

    for index, row in recipe_data.iterrows():
        recipe_ingredients = preprocess_text(row['Ingredients'])
        similarity_score_name = calculate_similarity(user_ingredients, preprocess_text(row['RecipeName']), vectorizer)
        similarity_score_translated = calculate_similarity(user_ingredients, preprocess_text(row['TranslatedRecipeName']), vectorizer)
        similarity_score_ingredients = calculate_similarity(user_ingredients, recipe_ingredients, vectorizer)

        if all(ingredient.strip() in recipe_ingredients.split(", ") for ingredient in user_ingredients_list):
            priority_recipes.append((row['RecipeName'], row['Ingredients'], row['Instructions'], 1.0))
        elif similarity_score_name > 0.5:
            priority_recipes.append((row['RecipeName'], row['Ingredients'], row['Instructions'], similarity_score_name))
        elif similarity_score_translated > 0.5:
            priority_recipes.append((row['RecipeName'], row['Ingredients'], row['Instructions'], similarity_score_translated))
        elif similarity_score_ingredients > 0.5:
            priority_recipes.append((row['RecipeName'], row['Ingredients'], row['Instructions'], similarity_score_ingredients))
        else:
            other_recipes.append((row['RecipeName'], row['Ingredients'], row['Instructions'], 0))

    priority_recipes.sort(key=lambda x: x[3], reverse=True)
    other_recipes.sort(key=lambda x: x[3], reverse=True)

    return priority_recipes, other_recipes

def generate_sequence_of_recommendations(priority_recipes, other_recipes):
    """Generates a sequence of recommended recipes."""
    sequence_recommendations = []
    item_number = 1

    for recipe, _, _, _ in priority_recipes:
        sequence_recommendations.append(f"{item_number}. {recipe}")
        item_number += 1

    for recipe, _, _, _ in other_recipes:
        sequence_recommendations.append(f"{item_number}. {recipe}")
        item_number += 1

    return sequence_recommendations

def recommendation_system(user_input, course_input, max_time_input, dataset_path):
    """Runs the recommendation system."""
    indian_recipe_data = pd.excel_file(dataset_path)

    user_input_processed = preprocess_text(user_input)
    indian_recipe_data['Ingredients'] = indian_recipe_data['Ingredients'].apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(indian_recipe_data['Ingredients'])

    priority_recipes, other_recipes = recommend_recipes(user_input_processed, indian_recipe_data, vectorizer)
    sequence_recommendations = generate_sequence_of_recommendations(priority_recipes, other_recipes)

    return sequence_recommendations, priority_recipes

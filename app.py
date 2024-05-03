import streamlit as st
from RecipeRecomSystem import recommendation_system  # Importing the recommendation system function from your logic file

# Title of the App
st.title("Recipe Recommendation System")

# User input fields
user_ingredients = st.text_input("Enter ingredients separated by commas")
course_options = ['Side Dish', 'Main Course', 'Breakfast', 'Lunch', 'Dinner', 'Snacks', 'Dessert']
selected_course = st.selectbox("Select Course", course_options)
max_cooking_time = st.number_input("Max Cooking Time (in minutes)", min_value=0)

# Initialize variables outside the button click block
selected_sequence_index = None
selected_item_index = None  # Initialize to None

# Button to trigger recommendation
if st.button("Get Recommendations"):
    # Set the dataset path
    dataset_path = 'C:/CapstoneProject/Datasets/IndianRecipeData.xlsx'  # Replace with your actual data path

    # Call the recommendation system function with user inputs
    sequence_recommendations, priority_recipes = recommendation_system(user_ingredients, selected_course, max_cooking_time,
                                                                        dataset_path)

    # Display the recommendations in a scrollable box
    if sequence_recommendations:
        st.write("Recommended Recipes:")
        recommended_recipes_text = "\n".join(sequence_recommendations)
        st.text_area(label="", value=recommended_recipes_text, height=400)

        # Input field for user to enter the sequence number of the item they like
        selected_item_index = st.number_input("Enter the sequence number of the item you like", min_value=1,
                                              max_value=len(priority_recipes) if priority_recipes else 0)

# Update UI Based on User Selection
if selected_item_index and selected_item_index <= len(priority_recipes):
    selected_recipe = priority_recipes[selected_item_index - 1]
    st.write(f"Detailed Recipe for {selected_recipe[0]}:")
    st.write(f"Ingredients: {selected_recipe[1]}")
    st.write(f"Instructions: {selected_recipe[2]}")
else:
    st.warning("Please select a valid item.")

# Footer
st.markdown("---")
st.markdown("Created by Team Id : 3559 "
            " Shivam Panchal")

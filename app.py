import streamlit as st
from RecipeRecomSystem import recommendation_system  # Importing the recommendation system function from your logic file

# Title of the App
st.title("Recipe Recommendation System")

# User input fields
user_ingredients = st.text_input("Enter ingredients separated by commas")
course_options = ['Side Dish', 'Main Course', 'Breakfast', 'Lunch', 'Dinner', 'Snacks', 'Dessert']
selected_course = st.selectbox("Select Course", course_options)
max_cooking_time = st.number_input("Max Cooking Time (in minutes)", min_value=0)

# Button to trigger recommendation
if st.button("Get Recommendations"):
    # Set the dataset path
    dataset_path = 'Datasets/IndianRecipeData.xlsx'  # Adjust the path as per your file structure

    # Call the recommendation system function with user inputs
    sequence_recommendations, _ = recommendation_system(user_ingredients, selected_course, max_cooking_time,
                                                        dataset_path)

    # Display the recommendations in a scrollable box
    if sequence_recommendations:
        st.write("Recommended Recipes:")
        recommended_recipes_text = "\n".join(sequence_recommendations)
        st.text_area(label="", value=recommended_recipes_text, height=400)

        # Input field for user to enter the sequence number of the item they like
        selected_item_index = st.number_input("Enter the sequence number of the item you like", min_value=1,
                                              max_value=len(sequence_recommendations))

        # Display detailed recipe for the selected item
        if st.button("Get Detailed Recipe"):
            selected_item_index -= 1  # Adjust for 0-based index
            selected_recipe = sequence_recommendations[selected_item_index]
            st.write(f"Detailed Recipe for {selected_recipe}:")  # Modify this to display the actual details
    else:
        st.write("No recommendations found.")

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
    # Call the recommendation system function with user inputs
    sequence_recommendations = recommendation_system(user_ingredients, selected_course, max_cooking_time)

    # Display the recommendations
    if sequence_recommendations:
        st.write("Recommended Recipes:")
        for item in sequence_recommendations:
            st.write(item)
    else:
        st.write("No recommendations found.")

# Footer
st.markdown("---")
st.markdown("Created by Your Name")

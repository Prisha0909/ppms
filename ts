def process_data(data):
    # Process the received data here
    st.write("Data received from Angular:", data)

# Define the main Streamlit app
def main():
    # Title for the Streamlit app
    st.title("Streamlit App")

    # Check if a POST request has been made
    if st.request_method() == 'POST':
        # Retrieve the data sent from Angular
        data = st.request_body()
        # Process the received data
        process_data(data)

# Run the Streamlit app
if __name__ == "__main__":
    main()

import streamlit as st

@st.experimental_memo
def process_pdf(file_content):
    st.markdown(file_content, unsafe_allow_html=True)  # Display PDF content

def main():
    st.title("PDF Viewer")

    # Create an endpoint to receive PDF content via GET request
    url = "/process_pdf"
    if st._is_running_with_streamlit:
        from streamlit.report_thread import get_report_ctx
        ctx = get_report_ctx()
        session_id = ctx.session_id
        url = st.server.get_url() + "/_api/session/" + session_id + "/process_pdf"

    file_content = st.text_area("Paste PDF content here")

    if file_content:
        process_pdf(file_content)

if __name__ == "__main__":
    main()

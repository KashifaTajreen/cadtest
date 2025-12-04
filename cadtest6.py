import streamlit as st

USERNAME = "student"
PASSWORD = "cad2026"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.title("AutoCAD Script Generator")

if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful")
        else:
            st.error("Invalid login")

else:
    st.subheader("Enter AutoCAD Question")

    question = st.text_area("Question", placeholder="PL_1: A triangular lamina of 30 mm sides resting on HP...")

    if st.button("Generate AutoCAD Script"):

        if not (question.startswith("P_") or question.startswith("L_") or question.startswith("PL_") or question.startswith("DLS_")):
            st.error("Invalid format! Must start with P_, L_, PL_, or DLS_")
        else:
            script = f"""
_LINE 0,0 100,0
_CIRCLE 50,40 20
_TEXT 10,80 2 0 {question}
"""
            with open("output.scr", "w") as f:
                f.write(script)

            with open("output.scr", "rb") as f:
                st.download_button("Download AutoCAD Script", f, file_name="output.scr")

    if st.button("Logout"):
        st.session_state.logged_in = False

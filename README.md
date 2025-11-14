## Initial Setup  
To set up your own Chatbot leveraging this Streamlit App, you first need to  
  (1) make sure Python is installed (v 3.10 or above), AND  
  (2) have access to an AI API (i.e. OpenAI, Snowflake, Claude, etc.)  

For preliminary setup, enter and run below in Windows Powershell (Mac users can also achieve this, but the code varies):  
**Create a work directory (only need to run once-when you are creating the folder/directory for the app)**  
  mkdir coursebot  
  
**Set directory to this folder**  
  cd coursebot  

  
**Create and activate a virtual environment (depending on your installation, "py" may be called "python" below)**  
  py -m venv .venv  

  .venv\Scripts\activate  

**Create three files we'll fill in next**  
  type NUL > app.py  
  type NUL > build_index.py  
  type NUL > requirements.txt  

**(Open requirements.txt and paste):**  
    streamlit  
    openai  
    faiss-cpu  
    tiktoken  
    pypdf  
    python-dotenv  

**Install these in Powershell**  
pip install -r requirements.txt

**Now, before you teach the API (one-time thing), make sure you have the training materials (pdf and .md files preferred) saved in**  
**the appropriate location listed in build_index.py. The current folder is named "course_materials" and it is located in the work directory**  
**"coursebot" created earlier.**   

**Next in Powershell, enter the API key. Below is example using OpenAI (update the key in quotes)**  
$env:OPENAI_API_KEY="YOUR_API_KEY_HERE"

**Now, you are ready to run the training**  
py build_index.py

**To test the app locally, in Powershell:**  
streamlit run app.py

**These files can be deployed to platforms such as Streamlit.**

**Side notes: if using OpenAI, you can cap the token/usage limits so that you don't get surprise bills**
**The specific AI model can be changed in app.py. Current model is specified to be gpt-4o-mini**

## When Updating the Streamlit Chatbot  
Oftentimes, we need to update the components and training materials for the chatbot. If you are simply updating the training materials, for example, follow the below routine. 

**In Powershell**  
cd coursebot  
py -m venv .venv  
.venv\Scripts\activate  
$env:OPENAI_API_KEY="YOUR_API_KEY_HERE"  
py build_index.py  

**(Optional) Test the chatbot locally**  
streamlit run app.py  

**Afterwards**  
Now, the API should have been "retrained". But you need to upload the updated **index.faiss** and **chunks.pkl** files to Github also. Don't forget this step!





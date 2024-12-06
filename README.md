After cloning this repository, run the following commands to install the necessary packages:
```
pip install fastapi uvicorn
pip install nltk
```
To install spaCy, run the following commands:
```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```
If your numpy version is 2.x instead of 1.x, this will be incompatible with spaCy, so address this by running this command:
```
pip install "numpy<2"
```
Inside the app.py file, uncomment line 12:
```
nltk.download('all')
```
And run app.py. This should show a successful package download, and once you have done this for the first time, you can recomment the line back out and leave it like that.

Next, to launch the server, run:
```
uvicorn app:app --reload
```

Once the server is launched, you can test making different API calls using test.py. Alternatively, you can see a visualization of your API calls using SwaggerUI at http://127.0.0.1:8000/docs#/default/

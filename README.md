# PDF

### key.png
여기에 논리도 올라와 있고 
<img width="556" alt="key" src="https://github.com/vlass-me/PDF/assets/104364766/c3ff1e8f-fe43-4526-8dd3-8159fcd2e6e4">

### embedding.ipynb
여기에 퀴즈 요약 등 프롬프트 만들어놓음

<img width="1092" alt="voice_summary_indexing" src="https://github.com/vlass-me/PDF/assets/104364766/2469b505-0505-4eec-a1a1-4bffa6832a8e">

<img width="851" alt="quiz" src="https://github.com/vlass-me/PDF/assets/104364766/235cb5ca-6665-4345-8e06-4c75de84f36e">


<img width="675" alt="pdf_extract" src="https://github.com/vlass-me/PDF/assets/104364766/2540254f-4854-46da-a8f9-41382f3bbe93">






Navigate to the app directory:

```
cd /path/to/chatgpt-retrieval-plugin
```

Install `poetry`:

```
pip install poetry
```

5. Create a new virtual environment:

```
poetry env use python3.10
```

6. Install the `retrieval-app` dependencies:

```
poetry install
```

7. Set app environment variables:

* `BEARER_TOKEN`: Secret token used by the app to authorize incoming requests. We will later include this in the request `headers`. The token can be generated however you prefer, such as using [jwt.io](https://jwt.io/).

* `OPENAI_API_KEY`: The OpenAI API key used for generating embeddings with the `text-embedding-ada-002` model. [Get an API key here](https://platform.openai.com/account/api-keys)!

8. Set Pinecone-specific environment variables:

* `DATASTORE`: set to `pinecone`.

* `PINECONE_API_KEY`: Set to your Pinecone API key. This requires a free Pinecone account and can be [found in the Pinecone console](https://app.pinecone.io/).

* `PINECONE_ENVIRONMENT`: Set to your Pinecone environment, looks like `us-east1-gcp`, `us-west1-aws`, and can be found next to your API key in the [Pinecone console](https://app.pinecone.io/).

* `PINECONE_INDEX`: Set this to your chosen index name. The name you choose is your choice, we just recommend setting it to something descriptive like `"openai-retrieval-app"`. *Note that index names are restricted to alphanumeric characters, `"-"`, and can contain a maximum of 45 characters.*

8. Run the app with:

```
poetry run start
```

If running the app locally you should see something like:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

In that case, the app has automatically connected to our index (specified by `PINECONE_INDEX`), if no index with that name existed beforehand, the app creates one for us.

Now we're ready to move on to populating our index with some data.


"""
환경변수 세팅 
export OPENAI_API_KEY =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
export PINECONE_API_KEY = 6757054f-e22e-4481-83a3-84b5ed0b0db8
export PINECONE_ENVIRONMENT = us-east1-gcp
export PINECONE_INDEX=pdf-test-data

"""

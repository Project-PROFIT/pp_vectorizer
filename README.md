# pp_vectorizer
This package provides a vectorizer (inherits from `sklearn.feature_extraction.text.TfidfVectorizer`) that uses a [PoolParty](poolparty.biz)
instance to add a semantic layer to the extraction. Namely, PoolParty enriches the resulting vector with extracted concepts (contained in the specified PoolParty project) and with additional information about those concepts.

To use:

1. Clone
2. ```pip3 install -r requirements.txt```
3. Use .env file to create a file called `.localenv` containing your credentials and configurations

   * *PP_USER*, *PP_PASSWORD* and *PP_SERVER* refer to your PoolParty credentials
   * *PP_PID*  Is the Project ID of the PoolParty project you wish to use for extraction
   * *CACHE_PATH*  Is a path where you want your local extraction temporary cache saved. /tmp would work
   * *STORE_PATH*  The extraction cache also has a long-term version, it will be saved here.
   * *DOCS_PATH*   Is a path where you have directories, each containing documents. Used for tests

4. Set your configuration by:  ```export $(cat .localenv | xargs)```
5. See examples directory and play around
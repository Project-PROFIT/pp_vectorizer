# pp_vectorizer
This package provides a vectorizer (inherits from `sklearn.feature_extraction.text.TfidfVectorizer`) that uses a [PoolParty](poolparty.biz) 
instance to add a semantic layer to the extraction. Namely, PoolParty enriches the resulting vector with extracted concepts (contained in the specified PoolParty project) and with additional information about those concepts.

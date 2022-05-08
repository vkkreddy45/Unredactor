import pytest
import unredactor

url = 'https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv'

def test_readdata():
    result = unredactor.Readdata(url)
    return result
    assert result is not None

r = test_readdata()

def test_preprocessing():
    result = unredactor.preprocess(r)
    return result
    assert result>0 and not None

def test_sentimentscore():
    result = unredactor.Sentimentscore(test_preprocessing())
    return result
    assert result is not None

def test_fngrams():
    result = unredactor.fngrams(test_sentimentscore())
    return result
    assert result == df

def test_vectorize():
    result = unredactor.Vectorize(test_fngrams())
    return result
    assert result == df3

def test_Prediction():
    result = unredactor.Prediction(test_vectorize())
    return result
    assert result is not None

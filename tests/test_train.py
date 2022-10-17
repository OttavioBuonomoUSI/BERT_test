from train import split_text

def test_split_text() -> int:
	assert( len(split_text("text"))==1 ) 
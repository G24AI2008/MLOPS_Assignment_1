from misc import run_training_pipeline
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
	
	print("__Running DecisionTreeRegressor Training Pipeline__")
run_training_pipeline(DecisionTreeRegressor)
from misc import run_training_pipeline
from sklearn.kernel_ridge import KernelRidge

if __name__ == "__main__":
	print("__Running KernelRidge Training Pipeline__")
	run_training_pipeline(KernelRidge)
# Price predictor
Stock price forecasting based on a Long Short Term Memory (LSTM) Recurrent Neural Network

*Browse Literature
	- LSTMs
	- LSTMs on financial data
	- LSTMs on Bitcoin data (Benchmarking)
	- Bitcoin (e. g. market behavior)
	- Tuneing Hyperparameters (EC)

*Hyperparameters:
parameter		initial
- Compression Window	60min=1h
- Sliding Window	5h
Hidden Layers (input and output excluded):
- Layer Depth		5 (1...7) (comment: there is a formula to calculate max. useful amount of layers ~ Tom, max. observed by Tom: 7)
- Layer Width		w = [a, b, c, d, e], len(w) == 5, a=b=c=d=e=2

- Activation Function	hidden neurons: [relU] - initial = relU, output neuron: identity
- initial weights: ??? check in lecture, they can be put in a smart way. read paper.

*Predict the compressed price for the next 60min
Take 5h (10-D) and predict the 6th hour's price (1-D)
	- Multiple linear regression in 10D to predicts 6h price
	Organize for this data matrix so that there is the 10 columns for t-6, t-5, t-4, t-3, t-1 (10-D), and then the price for t (1-D)

??? check if variance at every hidden layer is constant.

??? how to cope with the memory stuff





# Fine-Dust-Prediction
Prediction of daily average amount of fine dust by tensorflow.

## Preprocessing
Preprocessing daily average amount of fine dust(pm10), bundling it into 7 days.

For the convolutional layer, the amount of fine dust for each region is mapped to the geographical map of Seoul, 9x8 matrix.

<img src="md_image/seoul.png" width="50%">

The data was extracted from [Degree of Seoul Daily Average Air Pollution](http://data.seoul.go.kr/), provided by the Seoul Open Data Plaza.

## ConvLSTM Model

[Convolutional LSTM](https://arxiv.org/abs/1506.04214) is similar as LSTM, but it calculate hidden states by convolutional operation.

Feed 5D Tensor (time_dim, batch_size, height, width, channel) to ConvLSTM Cell and it returns hidden unit and state.

The `Predicator` model is the Stacked ConvLSTM Encoder-Decoder model. Stacked encoder summary the sequential data to the fixed-length vector and it is feeded to an initial state of the decoder model. Stacked decoder model generate sequential data and it is flattend by 1x1 convolution. It produces default 7 days prediction.

## Conclusion

Model was converged.

<img src="md_image/training_graph.png" width="40%">

In addition to the fine-dust, the accuracy can be improved by adding wind direction or date information.

*Prediction* <br>
<img src="md_image/prediction.png" width="40%">

*Forecaster* <br>
<img src="md_image/forecasting.png" width="40%">

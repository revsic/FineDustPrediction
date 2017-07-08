# Fine-Dust-Prediction
Prediction of the daily average amount of fine dust by tensorflow.

## Pre-processing
Pre-processing the daily average amount of the fine dust(pm10) to bundle it in 7 days.

For the convolution layer, the amount of fine dust for each region is mapped to the geographical map of Seoul, 9x8 matrix.

<img src="md_image/seoul.png" width="50%">

The data was extracted from the csv file [Seoul Daily Average Air Pollution Degree Information](http://data.seoul.go.kr/openinf/sheetview.jsp?infId=OA-2218&tMenu=11) provided by the Seoul Open Data Plaza.

## ConvLSTM Model

[Convolutional LSTM](https://arxiv.org/abs/1506.04214) is similar as LSTM, but it calculate hidden states by convolution not fully-connected.

Input 5D Tensor (input_dim, batch_size, height, width, channel) to the ConvLSTM Cell and it returns last hidden unit (batch_size, height, width, 32).

Flatten it by 1x1 convolution and calculate MSE loss with the next day's fine dust.

## Conclusion

Training it and model was converged.

<img src="md_image/training_graph.png" width="40%">

Unfortunately, learning did not work out better than I thought.

In addition to the fine-dust, the accuracy can be improved by adding wind direction or date information.

<img src="md_image/prediction.png" width="40%">

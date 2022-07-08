import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Initialize data (this is hard-coded)
data_cnn = [7.93, 8.68, 12.92, 17.24, 12.89, 25.67, 17.21, 17.27, 12.92, 12.86, 14.56]
data_rnn = [13.31, 19.95, 29.74, 29.84, 39.64, 39.71, 70.1, 30.24, 40.35, 50.05, 36.29]
data_rcnn = [31.2, 43.62, 60.73, 62.0, 103.06, 122.37, 82.54, 103.41, 163.71, 102.22, 87.49]
data_lstm = [35.31, 59.82, 88.8, 118.68, 148.61, 150.03, 90.38, 121.04, 119.72, 210.68, 114.31]

# data as np arrays
data_cnn_array = np.array(data_cnn)
data_rnn_array = np.array(data_rnn)
data_rcnn_array = np.array(data_rcnn)
data_lstm_array = np.array(data_lstm)

# Creates pandas DataFrame
index_values = ['ep1','ep2','ep3','ep4','ep5','ep6','ep7','ep8','ep9','ep10','avg']
column_values = ['t']

df_cnn = pd.DataFrame(data=data_cnn_array, index = index_values, columns = column_values)
#dr_tr_cnn=df_cnn.transpose()

df_rnn = pd.DataFrame(data=data_rnn_array, index = index_values, columns = column_values)
#dr_tr_rnn=df_rnn.transpose()

df_rcnn = pd.DataFrame(data=data_rcnn_array, index = index_values, columns = column_values)
#dr_tr_rcnn=df_rcnn.transpose()

df_lstm = pd.DataFrame(data=data_lstm_array, index = index_values, columns = column_values)
#dr_tr_lstm=df_lstm.transpose()

# plot time per epoch (exclude average in linegraph)
def time_graph():
    ax = df_cnn.iloc[0:10]['t'].plot(title="time per epoch")
    df_rnn.iloc[0:10]['t'].plot(ax=ax)
    df_rcnn.iloc[0:10]['t'].plot(ax=ax)
    df_lstm.iloc[0:10]['t'].plot(ax=ax)
    plt.legend(labels=["CNN","RNN","RCNN","LSTM"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("time (s)")
    plt.xticks(range(0,len(df_cnn.iloc[0:10].index)), df_cnn.iloc[0:10].index)
    plt.savefig("vis_time.png", dpi=300)
    plt.show()
    return ax

time_graph()


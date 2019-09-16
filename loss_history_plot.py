import plotly.graph_objs as go
import plotly.offline as py
import plotly
import plotly.io as pio
from IPython.display import display

class LossHistoryPlot(object):
    def __init__(self):
        super().__init__()
        traces = []
        traces.append(go.Scattergl(y=[], xaxis='x',  yaxis='y', 
                                   line={"color":'rgb(12, 12, 200)', "width":1}, mode = 'lines',))
        traces.append(go.Scattergl(y=[], xaxis='x',  yaxis='y',
                                   line={"color":'rgb(205, 12, 24)', "width":1}, mode = 'lines',))
        
        traces.append(go.Scattergl(y=[], xaxis='x2', yaxis='y2',
                                   line={"color":'rgb(12, 12, 200)', "width":1}, mode = 'lines',))
        traces.append(go.Scattergl(y=[], xaxis='x2', yaxis='y2',
                                   line={"color":'rgb(205, 12, 24)', "width":1}, mode = 'lines',))
        traces.append(go.Scattergl(y=[], xaxis='x2', yaxis='y2',
                                   line={"color":'rgb(12, 12, 200)', "width":0.5, "dash":"dashdot"}, mode = 'lines',))
        traces.append(go.Scattergl(y=[], xaxis='x2', yaxis='y2',
                                   line={"color":'rgb(205, 12, 24)', "width":0.5, "dash":"dashdot"}, mode = 'lines',))

        layout = plotly.tools.make_subplots(rows=1, cols=2, shared_xaxes=True, 
                                            print_grid=False,vertical_spacing=0.005).layout
        layout.height=500
        layout.showlegend=False
        self.fig = go.FigureWidget(
            data=traces, 
            layout=layout)
        display(self.fig)
        
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    
    def on_epoch_end(self, epoch, logs=None):
        """
        *logs* should be :
        {"loss":[..], "val_loss":[..],
         "acc":[..], "val_acc":[..],
         "f1_m":[..], "val_f1_m":[..]}       

        """
#         print(logs)
        self.fig.data[0].y += (logs["loss"],)
        self.fig.data[1].y += (logs["val_loss"],)
        self.fig.data[2].y += (logs["acc"],)
        self.fig.data[3].y += (logs["val_acc"],)
        self.fig.data[4].y += (logs["f1_m"],)
        self.fig.data[5].y += (logs["val_f1_m"],)
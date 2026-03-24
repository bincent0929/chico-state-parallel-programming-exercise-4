import plotly.graph_objects as go
import csv

def load_single_column_csv(path):
    values = []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if row:
                values.append(float(row[0]))
    return values

original = load_single_column_csv("Ex4-Acceleration-Profile.csv")
interpolated = load_single_column_csv("interop-output-10ps.csv")

# Original data is 1 sample/sec: t = 0, 1, 2, ..., 1800
orig_time = [float(i) for i in range(len(original))]

# Interpolated data is 10 samples/sec: t = 0, 0.1, 0.2, ..., 1800
steps_per_sec = (len(interpolated) - 1) // (len(original) - 1)
dt = 1.0 / steps_per_sec
interp_time = [i * dt for i in range(len(interpolated))]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=interp_time,
    y=interpolated,
    mode='lines',
    name=f'Interpolated ({steps_per_sec} samples/sec)',
    line=dict(color='royalblue', width=1.5),
))

fig.add_trace(go.Scatter(
    x=orig_time,
    y=original,
    mode='markers',
    name='Original (1 sample/sec)',
    marker=dict(color='red', size=4, symbol='circle'),
))

fig.update_layout(
    title='Acceleration Profile: Interpolated vs Original 1-Second Data',
    xaxis_title='Time (seconds)',
    yaxis_title='Acceleration',
    legend=dict(x=0.01, y=0.99),
    hovermode='x unified',
)

fig.show()

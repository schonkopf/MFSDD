import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class multifeature_dynamic_decomposition:
  def __init__(self, data, label):
    self.feature=data.columns.values
    self.data=np.array(data, dtype=float).T
    self.label=label
    
  def matrix_transformation(self, transformation='min-max', detrend_label=None):
    if transformation=='min-max':
      from sklearn.preprocessing import minmax_scale
      matrix = minmax_scale(self.data, axis=1)
    
    if detrend_label:
      import copy
      trend=copy.deepcopy(matrix)
      for n in range(matrix.shape[1]):
        trend[:,n]=np.mean(matrix[:,np.where(self.label[detrend_label]==self.label[detrend_label][n])[0]], axis=1)
      matrix=matrix-trend
    self.matrix=matrix

  def matrix_visualization(self, matrix, x, y, color_limit=None):
    if not color_limit:
      vmin=np.min(color_limit)
      vmax=np.max(color_limit)
    else:
      vmin=np.min(matrix)
      vmax=np.max(matrix)
      
    fig = go.Figure(data=go.Heatmap(z=matrix, x=x, y=y,
        colorscale='Jet', zmin = vmin, zmax = vmax))
    return fig

  def plot_2D_variation(self, data_reduced, color_label, groupby_label=None, disp='Summary'):
    y_range=np.max(data_reduced[:,1])-np.min(data_reduced[:,1])
    x_range=np.max(data_reduced[:,0])-np.min(data_reduced[:,0])
    y_range=[np.min(data_reduced[:,1])-y_range*0.1, np.max(data_reduced[:,1])+y_range*0.1]
    x_range=[np.min(data_reduced[:,0])-x_range*0.1, np.max(data_reduced[:,0])+x_range*0.1]
    data_reduced=pd.DataFrame(data_reduced)
    df=pd.concat([self.label, data_reduced], axis=1)
    
    if groupby_label:
      group_unique=np.unique(self.label[groupby_label])
      fig = make_subplots(rows=1, cols=len(group_unique), subplot_titles=group_unique.astype('object'))
      for n in range(len(group_unique)):
        fig=self.scatter_plot(fig, df, groupby_label, group_unique[n], color_label, disp=disp, row=1, col=n+1)

    else:
      fig = make_subplots(rows=1, cols=1)
      fig=self.scatter_plot(fig, df, groupby_label=None, name=None, color_label=color_label, disp=disp, row=1, col=1)
      
    fig.update_yaxes(range = y_range)
    fig.update_xaxes(range = x_range)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'white'})
    fig.update_xaxes(title='Dimension 1', showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(title='Dimension 2', showgrid=False, zeroline=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_traces(opacity=0.7)
    return fig
  
  def plot_dependence(self, factor_score, dependence_label, groupby_label=None, filter_size=1):
    from scipy.ndimage import uniform_filter1d
    factor_score=pd.DataFrame(factor_score)
    df=pd.concat([self.label, factor_score], axis=1)
    df=df.groupby(by=[groupby_label,dependence_label]).mean()
    df.reset_index(inplace=True)
    df2=pd.concat([self.label, factor_score], axis=1)
    df2=df2.groupby(by=[groupby_label,dependence_label]).std()
    df2.reset_index(inplace=True)

    subplot_titles=np.array(['Factor '+str(i+1) for i in range(factor_score.shape[1])])
    color=px.colors.qualitative.Pastel1

    fig = make_subplots(rows=int(np.ceil(factor_score.shape[1]/2)), cols=2, subplot_titles=subplot_titles, vertical_spacing=0.1)
    for j in range(factor_score.shape[1]):
      group_unique=np.unique(df[groupby_label])
      for i in range(len(group_unique)):
        ind=df[groupby_label]==group_unique[i]

        fig.add_trace(go.Scatter(x=df[dependence_label][ind], y=uniform_filter1d(df[j][ind]+df2[j][ind], size=filter_size), 
                    line=dict(color=color[i], width=0), showlegend=False, mode='lines', name=group_unique[i]), 
                    col=int(2-np.remainder(j+1,2)), row=int(np.floor(j/2)+1))
        fig.add_trace(go.Scatter(x=df[dependence_label][ind], y=uniform_filter1d(df[j][ind]-df2[j][ind], size=filter_size), 
                    line=dict(color=color[i], width=0), fill='tonexty', showlegend=False, mode='lines', name=group_unique[i]), 
                    col=int(2-np.remainder(j+1,2)), row=int(np.floor(j/2)+1))
        fig.add_trace(go.Scatter(x=df[dependence_label][ind], y=uniform_filter1d(df[j][ind], size=filter_size), 
                    line=dict(color=color[i]), mode='lines', name=subplot_titles[j]+'-'+group_unique[i]), 
                    col=int(2-np.remainder(j+1,2)), row=int(np.floor(j/2)+1))
        fig.update_yaxes(ticks="outside", tickwidth=2, showline=True, linewidth=2, linecolor='black', mirror=True, col=int(2-np.remainder(j+1,2)), row=int(np.floor(j/2)+1))
        fig.update_xaxes(ticks="outside", tickwidth=2, showline=True, linewidth=2, linecolor='black', mirror=True, col=int(2-np.remainder(j+1,2)), row=int(np.floor(j/2)+1))
        fig.update_yaxes(title='Factor score', col=int(2-np.remainder(j+1,2)), row=int(np.floor(j/2)+1))
        fig.update_xaxes(title=dependence_label, col=int(2-np.remainder(j+1,2)), row=int(np.floor(j/2)+1))

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'white'})
    return fig

  def subgroup(self, df, groupby_label):
    avg_x=df.groupby([groupby_label])[0].mean()
    avg_y=df.groupby([groupby_label])[1].mean()
    std_x=df.groupby([groupby_label])[0].std()
    std_y=df.groupby([groupby_label])[1].std()
    label=df.groupby([groupby_label])[groupby_label].mean()
    return avg_x, avg_y, std_x, std_y, label

  def scatter_plot(self, fig, df, groupby_label=None, name=None, color_label=None, disp='Summary', row=1, col=1):
    if disp=='Summary':
      if groupby_label:
        avg_x, avg_y, std_x, std_y, label=self.subgroup(df[df[groupby_label]==name], color_label)
      else:
        avg_x, avg_y, std_x, std_y, label=self.subgroup(df, color_label)
      fig.add_trace(go.Scatter(name=name, x=avg_x, y=avg_y, text=label, textposition='top center', 
            mode='markers+lines', marker=dict(color='DarkSlateGrey', size=1), line=dict(width=0.2, color='DarkSlateGrey'),
            error_y=dict(type='data', array=std_y, thickness=0.5, width=1),
            error_x=dict(type='data', array=std_x, thickness=0.5, width=1)),
            row=row, col=col)
      fig.add_trace(go.Scatter(name=name, x=avg_x, y=avg_y, text=label, textposition='top right', mode='markers+text',
            marker=dict(color=label, size=8)),
            row=row, col=col)
    elif disp=='Raw':
      if groupby_label:
        x=df[df[groupby_label]==name][0]
        y=df[df[groupby_label]==name][1]
        label=df[df[groupby_label]==name][color_label]
      else:
        x=df[0]
        y=df[1]
        label=df[color_label]
      fig.add_trace(go.Scatter(name=name, x=x, y=y, mode='markers',
            marker=dict(color=label, size=8, colorbar=dict(thickness=10))),
            row=row, col=col)
    return fig

  def sunburst(self, W, H, expression=None, reconstruct_source=None, H_supplementary=[]):
    contribution=np.zeros(H.shape[1])
    reconstruction = np.dot(W, H)
    for n in range(H.shape[1]):
      source = np.dot(W[:,n:n+1], H[n:n+1,:])
      contribution[n]=np.sum(source)/np.sum(reconstruction)
    contribution=contribution/np.sum(contribution)
    
    W_cluster=np.argmax(W, axis=1)
    W_max=W[np.arange(W.shape[0]),W_cluster]
    for n in range(H.shape[1]):
      W_max[W_cluster==n]=contribution[n]*(W_max[W_cluster==n]/np.sum(W_max[W_cluster==n]))
    if type(expression) == type(None):
      np.seterr(divide='ignore',invalid='ignore')
      expression=np.nanmean(np.multiply(self.matrix, np.divide(np.dot(W[:,reconstruct_source], H[reconstruct_source,:]),reconstruction)),axis=1)
      
    if len(H_supplementary)>0:
      df = pd.DataFrame({'Source':['Factor '+str(i+1)+' ('+H_supplementary[i]+')' for i in W_cluster],
              'Contribution':W_max, 'Expression':expression})
    else:
      df = pd.DataFrame({'Source':['Factor '+str(i+1) for i in W_cluster],
              'Contribution':W_max, 'Expression':expression})
    df=pd.concat([df, self.feature], axis=1)

    fig = px.sunburst(df, path=np.insert(self.feature.columns,0,'Source'), values='Contribution', 
                      color='Expression', color_continuous_scale='haline')
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'white'})
    return fig

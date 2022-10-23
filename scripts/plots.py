import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from IPython.display import Image
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go  
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from IPython.display import Markdown, display, Image, display_html
from causalnex.structure.notears import from_pandas


from logger import get_logger 
my_logger = get_logger("Plot")
my_logger.debug("Loaded successfully!")

class Plots:
    def __init__(self) -> None:
        pass

    def hist(self, df:pd.DataFrame, column:str, color:str)->None:
        plt.figure(figsize=(9, 7))
        sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()


    def box_plot(self, df: pd.DataFrame, x_col: str, title: str)->None:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.show()


    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(0, 10000)
        plt.ylim(0, 10000)
        plt.show()


    def heatmap(self, df: pd.DataFrame, title: str, cbar=False) -> None:
        correlation = df.corr()
        plt.figure(figsize=(20, 17))
        sns.heatmap(correlation, annot=True, cmap='viridis', vmin=0,
                    vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
        plt.title(title, size=18, fontweight='bold')
        plt.show()
         

    def plot_bar(self, column, title, xlabel, ylabel):
        plt.figure(figsize=(10,7))
        sns.barplot(y=column.index, x=column.values) 
        plt.title(title, size=14, fontweight="bold")
        plt.xlabel(xlabel, size=13, fontweight="bold") 
        plt.ylabel(ylabel, size=13, fontweight="bold")
        plt.show() 

    def mult_hist(self, sr, rows, cols, title_text, subplot_titles, interactive=False):
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
        for i in range(rows):
            for j in range(cols):
                x = ["-> " + str(i) for i in sr[i+j].index]
                fig.add_trace(go.Bar(x=x, y=sr[i+j].values), row=i+1, col=j+1)
        fig.update_layout(showlegend=False, title_text=title_text)
        if(interactive):
            fig.show()
        else:
            return Image(pio.to_image(fig, format='png', width=1200))


    def plot_hist(self, df: pd.DataFrame, column: str, color: str) -> None:
        plt.figure(figsize=(9, 7))
        sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()


    def feature_vs_target(self, df, x):    

        fig,ax = plt.subplots(nrows = 10, ncols = 3, figsize = (12,24),dpi=80)
        axes = ax.ravel()

        for col,ax in zip(x.columns,axes):
            # plots
            sns.kdeplot(df[col], ax = ax, shade = True ,
                        palette=["red", "green"],
                        alpha = 0.5, linewidth = 1, ec = 'black',
                        hue = df['diagnosis'], hue_order = ['M','B'],
                        legend = False)

            # plot setting
            xlabel = ' '.join([value.capitalize() for value in str(col).split('_') ])
            ax.axes.set_xlabel(xlabel,{'font':'serif','size':10, 'weight':'bold'}, alpha = 1)

        plt.tight_layout(pad= 2,h_pad = 1, w_pad = 1)

        fig.text(0.615,1, "\n       Benign",{'font':'serif','size':14, 'weight':'bold', 'color':"green"}, alpha = 1)
        fig.text(0.735,1, '|',{'font':'serif','size':16, 'weight':'bold'})
        fig.text(0.75,1, "  Malignant",{'font':'serif','size':14, 'weight':'bold','color':"red"}, alpha = 1)

        fig.show()


    def multiple_boxplot(self, x, y, start: int = 0, num_features: int = 10):
        data = pd.concat([y, x.iloc[:, start:num_features]], axis=1)
        data = pd.melt(data,
                    id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
        plt.figure(figsize=(20, 12))
        sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
        plt.xticks(rotation=90)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()


    def vis_sm(self, sm):
        viz = plot_structure(
            sm,
            graph_attributes={"scale": "2.0", 'size': 2.5},
            all_node_attributes=NODE_STYLE.WEAK,
            all_edge_attributes=EDGE_STYLE.WEAK)
        return Image(viz.draw(format='png'))

    
    def causal_graph (self,data, parent_node, percent):
        """Draws Causal graph
        Args:
            structural_model (from_pandas_lasso): Structural model of causalnex
        Returns:
            plot_structure
        """
        try:
            portion = int(data.shape[0] * percent)
            x_portion = data.head(portion)

            sm = from_pandas(x_portion, tabu_parent_nodes=[parent_node],)
            sm.remove_edges_below_threshold(0.8)
            sm = sm.get_largest_subgraph() 
            return sm
        
        except Exception:
            self.logger.exception('plots causal graph  failed.')

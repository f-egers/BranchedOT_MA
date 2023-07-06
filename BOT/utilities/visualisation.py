import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def visualize_solution(topo, coords, EW, bot_problem_dict, show=True, save=False, save_name=None, alt_node_color=None, title="", ax=None, edge_color="black", edge_alpha=0.75, plot_BP=True, edge_flows="linewidth",scale_terminals=True,draw_legend=True):
    
    """
    Visualizes the solution for a Branched Transportation Problem.

    Parameters:
    - topo (ndarray or topology): Topology of the transportation network.
    - coords (ndarray): Coordinates of the nodes in the network.
    - EW (ndarray): Edge weights representing the flows in the network.
    - bot_problem_dict (dict): Dictionary containing problem information.
    - show (bool): Whether to display the plot (default: True).
    - save (bool): Whether to save the plot (default: False).
    - save_name (str): Name of the file to save the plot (default: None).
    - alt_node_color (list or None): Alternate colors for nodes (default: None).
    - title (str): Title for the plot (default: "").
    - ax (matplotlib Axes or None): Axes object to plot on (default: None).
    - edge_color (str): Color of the edges (default: "black").
    - edge_alpha (float): Alpha value for edge transparency (default: 0.75).
    - plot_BP (bool): Whether to plot the Branching Points (default: True).
    - edge_flows (str): Method to represent edge flows (default: "linewidth").
    - scale_terminals (bool): Whether to scale terminal sizes based on supply/demand (default: True).
    - draw_legend (bool): Whether to draw a legend for the plot (default: True).

    Returns:
    - fig (matplotlib Figure): Figure object containing the plot.
    """

    if hasattr(topo,"adj"):
        adj = topo.adj
    else:
        adj = topo

    #unpack problem dict
    al = bot_problem_dict["al"]
    coords_sources = bot_problem_dict["coords_sources"]
    coords_sinks = bot_problem_dict["coords_sinks"]
    supply_arr = bot_problem_dict["supply_arr"]
    demand_arr = bot_problem_dict["demand_arr"]

    flows = np.abs(EW)

    n = int(len(coords_sinks)+len(coords_sources))#int(len(coords)/2+1)

    if len(supply_arr)==0:
        flow_scale_fact = np.max(flows)
    elif np.abs(np.sum(supply_arr)-np.sum(demand_arr)) > 1e-2:
        flow_scale_fact = np.max(flows)
    else:
        flow_scale_fact = sum(supply_arr)
    linescale = 15 / flow_scale_fact  # defines thickness of edges relative to total flow
    markerscale = 25 / flow_scale_fact

    #with plt.style.context("ggplot"):

    if ax is None:
        fig = plt.figure(figsize=(8*(np.max(coords[:,0])-np.min(coords[:,0])), 8*(np.max(coords[:,1])-np.min(coords[:,1]))))
        ax = fig.gca()

    #plot edges
    for i,a in enumerate(adj):
        na=i+n
        for j,nb in enumerate(a):
            if na>nb:
                flow = flows[i][j]
                if edge_flows == "linewidth":
                    if flow == 0:
                        ax.plot([coords[na][0], coords[nb][0]], [coords[na][1], coords[nb][1]], color=edge_color, linewidth=1, linestyle="dashed", alpha=edge_alpha, solid_capstyle='round')
                    else:
                        ax.plot([coords[na][0], coords[nb][0]], [coords[na][1], coords[nb][1]], color=edge_color, linewidth=linescale * flow + 1, alpha=edge_alpha, solid_capstyle='round')
                else:
                    if flow == 0:
                        ax.plot([coords[na][0], coords[nb][0]], [coords[na][1], coords[nb][1]], color=edge_color, linewidth=2, linestyle="dashed", alpha=edge_alpha, solid_capstyle='round')
                    else:
                        ax.plot([coords[na][0], coords[nb][0]], [coords[na][1], coords[nb][1]], color=edge_color, linewidth=2, alpha=edge_alpha, solid_capstyle='round')
                    

    #plot BPs
    if plot_BP:
        for bp,fl in enumerate(flows):
            ax.plot(coords[bp+n][0], coords[bp+n][1], marker="o", color=edge_color, linestyle="", markersize=linescale*np.max(fl)+1,alpha=1)

    if alt_node_color is not None:
        colors = alt_node_color
    else:
        colors = ["r"]*len(supply_arr)
        colors.extend(["b"]*len(demand_arr))

    #plot terminals:
    if scale_terminals:
        for s,d,c in zip(coords_sinks,demand_arr,colors[len(supply_arr):]):
            ax.plot(s[0], s[1], marker="o", color=c, linestyle="", markersize=markerscale * d + 3, alpha=1.0, label="sinks")
        for s,d,c in zip(coords_sources,supply_arr,colors[:len(supply_arr)]):
            ax.plot(s[0], s[1], marker="o", color=c, linestyle="", markersize=markerscale * d + 3, alpha=1.0, label="sources")
    else:
        for s,d,c in zip(coords_sinks,demand_arr,colors[len(supply_arr):]):
            ax.plot(s[0], s[1], marker="o", markersize = 4, color=c, linestyle="", alpha=1.0, label="sinks")
        for s,d,c in zip(coords_sources,supply_arr,colors[:len(supply_arr)]):
            ax.plot(s[0], s[1], marker="o", markersize = 6, color=c, linestyle="", alpha=1.0, label="sources")


    ax.axis('equal')
    ax.set_title(title,loc="left")
    if alt_node_color is None:
        legend_patches = [pat.Circle((0,0),radius=5,fill=True,color="r",label="sources"),pat.Circle((0,0),radius=5,fill=True,color="b",label="sinks")]
        if draw_legend:
            legend = ax.legend(handles=legend_patches)
    if save:
        #plt.xticks([])
        #plt.yticks([])
        if save_name is None:
            save_name="BOT_solution"
        plt.savefig(save_name + ".pdf", bbox_inches="tight")
    if show:
        plt.show()
    #else:
    #    plt.close()
    return ax.figure

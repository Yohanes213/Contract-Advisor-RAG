import plotly.graph_objects as go

def visualize_result(result, name, title):
    """
    Creates a radar chart to visualize various performance metrics.

    Parameters:
    result (dict): A dictionary containing the performance metrics.
        Keys should include 'context_precision', 'context_recall', 
        'faithfulness', 'answer_relevancy', and 'answer_similarity'.
    name (str): The name of the output HTML file where the chart will be saved.
    title (str): The title of the radar chart.

    Returns:
    None: The function saves the radar chart as an HTML file.
    """

    data = {
        'context_precision': result['context_precision'],
        'context_recall': result['context_recall'],
        'faithfulness': result['faithfulness'],
        'answer_relevancy': result['answer_relevancy'],
        'answer_similarity': result['answer_similarity']
    }

    # Create the radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(data.values()),
        theta=list(data.keys()),
        fill='toself',
        name='Ensemble Rag'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title=title,
        width=800
    )

    # Save the figure as an HTML file
    fig.write_html(name)

import plotly.graph_objects as go

def visualize_result(result, name):

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
        title='Retrieval Augmented Generation - Evaluation',
        width=800
    )

    # Save the figure as an HTML file
    fig.write_html(name)

